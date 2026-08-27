"""
Session Manager for Claude Agent SDK.

Manages Claude SDK sessions with full hook support and streaming responses.
Includes:
- Dangerous command blocking (security)
- AskUserQuestion support (interactive sessions)
- Auto-compact handling (context management)
- Session continuation support (long-running tasks)
- PostToolUse audit logging (observability)
- Output persistence (session recovery)
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import uuid
import logging
import json
import re
import time
import subprocess
import shutil

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    HookMatcher,
    HookContext,
)

logger = logging.getLogger(__name__)

# Fix(TAS-624): a turn boundary marker placed on a session's queue by the
# persistent receive loop. `send_prompt` returns when it drains one, which is
# what gives the RPC per-turn semantics -- the rest of the chain already assumes
# them. A distinct type rather than None so a future None message cannot be
# mistaken for the end of a turn.
#
# Fix(TAS-808): the marker is TAGGED with which turn it closes, because a bare
# sentinel made correctness depend on `asyncio.Queue` waking getters in FIFO
# order -- load-bearing, and documented nowhere near the call site.
#
# Interrupting a turn makes the CLI emit its ResultMessage ~1.45s LATER
# (measured), while `interrupt()` marks the session idle at once. So a fused
# continuation is dispatched, and starts reading, before the interrupted turn's
# marker exists. Normally FIFO saves it: the interrupted RPC has been parked
# longer and takes the stale marker. But when that RPC is gone -- client
# disconnected, WebSocket dropped, caller cancelled -- the continuation is the
# only getter. It took the stale marker and returned IMMEDIATELY WITH NO
# OUTPUT: prompt accepted, nothing back, session idle, no error anywhere.
#
# Turns are paired by sequence: the k-th prompt on a session is closed by the
# k-th ResultMessage, because the CLI emits exactly one per prompt (including
# on interrupt). A reader that drains a marker older than its own drops it and
# keeps waiting.
class _TurnEnd:
    """End of one turn. `seq` is None for an unconditional release.

    An unconditional marker is used where the turn can no longer complete at
    all -- the transport died, or the session was closed -- so whoever is
    waiting must be released regardless of which turn they were serving.
    Otherwise they would wait for an end that can never arrive, which is the
    never-ending turn TAS-624 removed.
    """

    __slots__ = ("seq",)

    def __init__(self, seq: Optional[int] = None) -> None:
        self.seq = seq

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_TurnEnd(seq={self.seq})"

# Fix(TAS-624): how many undrained messages a session's queue keeps before it
# stops retaining more of the CURRENT turn. The queue is deliberately UNBOUNDED
# so the receive loop can never block on `put` -- blocking is the exact failure
# this fix exists to remove, because the SDK services its 100-slot message
# stream from the same task that routes control_response frames and dispatches
# hook callbacks. Once that task parks, interrupt(), set_model() and every
# PreToolUse hook stop being answered, with no error anywhere.
#
# So the cap is enforced by DROPPING rather than by back-pressure, and only for
# a reader that has gone away mid-turn (a cancelled RPC while the CLI keeps
# producing). Dropped messages are still written to _session_outputs, which is
# the durable record `save_session_outputs`/`load_session_outputs` use -- the
# queue is only the live view. A turn boundary is never dropped; losing one
# would hang the next RPC forever.
_MAX_UNDRAINED_MESSAGES = 2000


@dataclass
class SessionCredentials:
    """Ephemeral credentials for a session.

    These credentials are used only for the duration of the session
    and are cleaned up when the session ends. They can be re-injected
    for subsequent sessions.
    """
    credential_type: str  # "api_key" or "oauth"
    api_key: Optional[str] = None
    oauth_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None  # Unix timestamp


@dataclass
class SessionConfig:
    """Configuration for creating a new Claude session."""
    project_path: str
    model: str = "sonnet"
    permission_mode: str = "default"  # "default", "bypassPermissions", "acceptEdits"
    allowed_tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)
    max_turns: Optional[int] = None
    continue_conversation: bool = False
    environment: Dict[str, str] = field(default_factory=dict)
    enable_dangerous_command_blocking: bool = True
    headless: bool = True
    account_id: Optional[str] = None
    user_email: Optional[str] = None
    # Ephemeral session credentials - cleaned up when session ends
    credentials: Optional[SessionCredentials] = None
    # Settings file path - if provided, loads settings from this path
    # If not provided, server may load from default container mount location
    settings_path: Optional[str] = None


@dataclass
class SessionInfo:
    """Information about an active Claude session."""
    id: str
    name: str
    status: str  # "pending", "running", "idle", "completed", "error", "terminated", "waiting_for_input"
    project_path: str
    model: str
    created_at: datetime
    sdk_session_id: Optional[str] = None
    context_tokens: int = 0
    max_context_tokens: int = 200000
    continuation_count: int = 0
    summary: Optional[str] = None
    account_id: Optional[str] = None
    has_ephemeral_credentials: bool = False  # Track if session has injected credentials
    user_email: Optional[str] = None


@dataclass
class StreamMessage:
    """Unified message type for streaming responses."""
    type: str  # "text", "thinking", "tool_use", "tool_result", "cost", "error", "status", "question", "blocked_command", "subagent_complete"
    content: Optional[str] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    is_error: bool = False
    cost_usd: Optional[float] = None
    status: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Question fields
    question_id: Optional[str] = None
    question_text: Optional[str] = None
    question_options: Optional[List[Dict[str, str]]] = None
    allow_custom: bool = False
    # Blocked command fields
    blocked_category: Optional[str] = None
    blocked_pattern: Optional[str] = None
    # Subagent fields
    agent_id: Optional[str] = None
    agent_duration_ms: Optional[float] = None


@dataclass
class QuestionData:
    """Data for a pending AskUserQuestion."""
    question_id: str
    questions: List[Dict[str, Any]]
    tool_use_id: str


@dataclass
class ToolExecution:
    """Record of a tool execution for audit logging."""
    tool_use_id: str
    tool_name: str
    input_data: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    result: Optional[str] = None
    is_error: bool = False
    was_blocked: bool = False


@dataclass
class SessionOutput:
    """A single output message from a session (for persistence)."""
    timestamp: str
    message_type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AccountCredentials:
    """Stored credentials for a Claude account."""
    account_id: str
    credential_type: str  # "api_key" or "oauth"
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None  # Unix timestamp
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class SubagentInfo:
    """Information about a running subagent (Task tool)."""
    agent_id: str
    session_id: str
    start_time: float
    prompt: Optional[str] = None
    status: str = "running"  # "running", "completed", "error"


class SessionManager:
    """
    Manages Claude SDK sessions with full hook support.

    This is the core integration layer between gRPC and the Claude Agent SDK.
    It handles session lifecycle, streaming responses, hook callbacks,
    dangerous command blocking, and AskUserQuestion support.
    """

    # Dangerous command patterns (ported from tascim-web)
    DANGEROUS_PATTERNS = [
        # Filesystem destruction
        (r'\brm\s+-rf\s+(/|~|\$HOME)', "FILESYSTEM_DESTRUCTION"),
        (r'\bdd\s+.*of=/dev/', "FILESYSTEM_DESTRUCTION"),
        (r'\b(:|true)\s*>\s*/dev/sd', "FILESYSTEM_DESTRUCTION"),
        (r'\bmkfs\.', "FILESYSTEM_DESTRUCTION"),
        (r'\bchmod\s+-R\s+777\s+/', "PERMISSION_CHANGE"),
        (r'\bchown\s+-R\s+.*\s+/', "PERMISSION_CHANGE"),
        (r'>\s*/etc/(passwd|shadow|sudoers)', "AUTH_FILE_MODIFICATION"),
        (r'\bcurl\s+.*\|\s*(ba)?sh', "REMOTE_CODE_EXECUTION"),
        (r'\bwget\s+.*\|\s*(ba)?sh', "REMOTE_CODE_EXECUTION"),

        # Git force push
        (r'\bgit\s+push\s+.*--force\b', "GIT_FORCE_PUSH"),
        (r'\bgit\s+push\s+.*-f\b', "GIT_FORCE_PUSH"),
        (r'\bgit\s+push\s+--force', "GIT_FORCE_PUSH"),
        (r'\bgit\s+push\s+-f\s', "GIT_FORCE_PUSH"),
        (r'\bgit\s+push\s+.*--force-with-lease', "GIT_FORCE_PUSH"),

        # Git destructive operations
        (r'\bgit\s+reset\s+--hard\s+origin/', "GIT_DESTRUCTIVE"),
        (r'\bgit\s+clean\s+-fd', "GIT_DESTRUCTIVE"),
        (r'\bgit\s+checkout\s+--\s+\.', "GIT_DESTRUCTIVE"),

        # Git history rewriting
        (r'\bgit\s+rebase\s+.*--exec', "GIT_HISTORY_REWRITE"),
        (r'\bgit\s+filter-branch', "GIT_HISTORY_REWRITE"),
        (r'\bgit\s+filter-repo', "GIT_HISTORY_REWRITE"),
    ]

    def __init__(self):
        self._sessions: Dict[str, Tuple[SessionInfo, SessionConfig, Optional[ClaudeSDKClient]]] = {}
        self._startup_time = datetime.now(timezone.utc)

        # Question handling
        self._pending_questions: Dict[str, QuestionData] = {}  # session_id -> QuestionData
        self._question_responses: Dict[str, asyncio.Queue] = {}  # session_id -> response queue
        self._waiting_for_answer: Dict[str, asyncio.Event] = {}  # session_id -> event

        # Tool execution tracking (for PostToolUse audit logging)
        self._tool_executions: Dict[str, ToolExecution] = {}  # tool_use_id -> ToolExecution
        self._session_tool_history: Dict[str, List[ToolExecution]] = {}  # session_id -> [ToolExecution]

        # Output persistence
        self._session_outputs: Dict[str, List[SessionOutput]] = {}  # session_id -> [SessionOutput]
        self._output_dir = Path.home() / ".claude" / "state" / "session_outputs"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Audit logs
        self._audit_log_path = Path.home() / ".claude" / "state" / "logs" / "blocked_commands.log"
        self._tool_audit_log_path = Path.home() / ".claude" / "state" / "logs" / "tool_executions.log"
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Account/OAuth management
        self._accounts: Dict[str, AccountCredentials] = {}  # account_id -> AccountCredentials
        self._accounts_file = Path.home() / ".claude" / "state" / "grpc_accounts.json"
        self._load_accounts()

        # Subagent tracking (for SubagentStop hook)
        self._subagents: Dict[str, SubagentInfo] = {}  # agent_id -> SubagentInfo

        # Fix(TAS-624): one persistent consumer of the SDK stream per session,
        # feeding a queue that each SendPrompt RPC drains for the length of one
        # turn. See _receive_loop for why the consumer must outlive the RPC.
        self._receive_tasks: Dict[str, asyncio.Task] = {}  # session_id -> consumer task
        self._turn_queues: Dict[str, asyncio.Queue] = {}   # session_id -> live message queue
        # Fix(TAS-808): the k-th prompt is closed by the k-th ResultMessage.
        # Kept separately because they legitimately diverge for ~1.45s while an
        # interrupted turn's result is still in flight -- that gap IS the bug.
        self._prompt_seq: Dict[str, int] = {}   # session_id -> prompts sent
        self._result_seq: Dict[str, int] = {}   # session_id -> results seen
        self._dropped_messages: Dict[str, int] = {}        # session_id -> count dropped this turn

        # Find Claude CLI path for OAuth operations
        self._cli_path = self._find_cli()

        # Ephemeral session credentials tracking
        # Maps session_id -> original env values to restore on cleanup
        self._ephemeral_credentials: Dict[str, Dict[str, Optional[str]]] = {}

        logger.info("SessionManager initialized with full audit logging, output persistence, and account management")

    async def create_session(self, config: SessionConfig) -> SessionInfo:
        """Create a new Claude session."""
        session_id = str(uuid.uuid4())
        project_name = config.project_path.split("/")[-1] or "unknown"
        session_name = f"{project_name}_{session_id[:6]}"

        has_ephemeral_credentials = False
        credentials_to_use = config.credentials

        # Load settings from file if provided or from default location
        settings_path = config.settings_path or self._get_default_settings_path()
        if settings_path and not credentials_to_use:
            settings = self._load_settings_from_path(settings_path)
            if settings:
                # Extract credentials from settings
                credentials_to_use = self._extract_credentials_from_settings(settings)
                if credentials_to_use:
                    logger.info(f"Session {session_id}: Loaded credentials from settings file: {settings_path}")

        # Fallback: Try to load from mounted .claude folder (host credential sharing)
        if not credentials_to_use:
            credentials_to_use = self._get_cli_credentials_from_mount()
            if credentials_to_use:
                logger.info(f"Session {session_id}: Loaded credentials from mounted .claude folder")

        # Set up ephemeral session credentials if provided or loaded from settings
        if credentials_to_use:
            self._inject_ephemeral_credentials(session_id, credentials_to_use)
            has_ephemeral_credentials = True
            logger.info(f"Session {session_id}: Ephemeral {credentials_to_use.credential_type} credentials injected")
        # Otherwise, set up account environment if account_id provided
        elif config.account_id:
            await self._setup_account_env(config.account_id)

        session_info = SessionInfo(
            id=session_id,
            name=session_name,
            status="idle",
            project_path=config.project_path,
            model=config.model,
            created_at=datetime.now(timezone.utc),
            account_id=config.account_id,
            has_ephemeral_credentials=has_ephemeral_credentials,
            user_email=config.user_email,
        )

        # Initialize question handling for this session
        self._question_responses[session_id] = asyncio.Queue()
        self._waiting_for_answer[session_id] = asyncio.Event()
        self._waiting_for_answer[session_id].set()  # Start in non-blocking state

        # Initialize tool tracking and output persistence for this session
        self._session_tool_history[session_id] = []
        self._session_outputs[session_id] = []

        self._sessions[session_id] = (session_info, config, None)
        logger.info(f"Session created: {session_id} ({session_name}) for {config.project_path}")
        return session_info

    def _inject_ephemeral_credentials(
        self,
        session_id: str,
        credentials: SessionCredentials
    ) -> None:
        """Inject ephemeral credentials into environment for a session.

        Stores the original environment values so they can be restored
        when the session ends.
        """
        import os

        # Store original values for cleanup
        original_values: Dict[str, Optional[str]] = {}

        if credentials.credential_type == "api_key" and credentials.api_key:
            # Anthropic API key
            original_values["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
            os.environ["ANTHROPIC_API_KEY"] = credentials.api_key
            logger.debug(f"Session {session_id}: Set ANTHROPIC_API_KEY from ephemeral credentials")

        elif credentials.credential_type == "oauth":
            # OAuth token for Claude Code
            if credentials.oauth_token:
                original_values["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
                os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = credentials.oauth_token
                logger.debug(f"Session {session_id}: Set CLAUDE_CODE_OAUTH_TOKEN from ephemeral credentials")

            if credentials.refresh_token:
                original_values["CLAUDE_CODE_REFRESH_TOKEN"] = os.environ.get("CLAUDE_CODE_REFRESH_TOKEN")
                os.environ["CLAUDE_CODE_REFRESH_TOKEN"] = credentials.refresh_token

        self._ephemeral_credentials[session_id] = original_values

    def _cleanup_ephemeral_credentials(self, session_id: str) -> None:
        """Clean up ephemeral credentials when a session ends.

        Restores original environment values.
        """
        import os

        original_values = self._ephemeral_credentials.pop(session_id, None)
        if not original_values:
            return

        for key, original_value in original_values.items():
            if original_value is None:
                # Remove the env var if it wasn't set before
                os.environ.pop(key, None)
            else:
                # Restore original value
                os.environ[key] = original_value

        logger.info(f"Session {session_id}: Ephemeral credentials cleaned up")

    def _load_settings_from_path(self, settings_path: str) -> Optional[Dict[str, Any]]:
        """Load settings from a file path.

        Supports JSON settings files. The settings can contain credentials
        and other configuration that will be applied to the session.

        Args:
            settings_path: Path to the settings file

        Returns:
            Dict containing settings, or None if file not found/invalid
        """
        path = Path(settings_path)
        if not path.exists():
            logger.warning(f"Settings file not found: {settings_path}")
            return None

        try:
            with open(path, "r") as f:
                settings = json.load(f)
            logger.info(f"Loaded settings from {settings_path}")
            return settings
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in settings file {settings_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load settings from {settings_path}: {e}")
            return None

    def _get_default_settings_path(self) -> Optional[str]:
        """Get the default settings path for container/microservice environments.

        Checks common mount locations for settings files:
        - /config/claude-settings.json (container mount)
        - /etc/claude/settings.json (system config)
        - /mnt/claude/settings.json (shared host mount for containers)
        - ~/.claude/settings.json (user config)
        - /run/secrets/claude-settings (Docker/Kubernetes secrets)

        Returns:
            Path to the first found settings file, or None
        """
        default_locations = [
            Path("/config/claude-settings.json"),
            Path("/etc/claude/settings.json"),
            # Shared .claude folder mount (for containers with host credential sharing)
            Path("/mnt/claude/settings.json"),
            Path("/mnt/host-claude/settings.json"),
            Path.home() / ".claude" / "settings.json",
            Path("/run/secrets/claude-settings"),
            Path("/var/run/secrets/claude/settings.json"),
        ]

        for location in default_locations:
            if location.exists() and location.is_file():
                logger.info(f"Found default settings file: {location}")
                return str(location)

        return None

    def _get_cli_credentials_from_mount(self) -> Optional[SessionCredentials]:
        """Try to load CLI credentials from a mounted .claude folder.

        When credentials are not explicitly set, this method checks for
        a shared mount of the host's .claude folder which contains CLI
        authentication state.

        Mount locations checked:
        - /mnt/claude/ (standard container mount)
        - /mnt/host-claude/ (alternative mount)
        - ~/.claude/ (fallback to user home)

        Returns:
            SessionCredentials if CLI auth state found, None otherwise
        """
        mount_locations = [
            Path("/mnt/claude"),
            Path("/mnt/host-claude"),
            Path.home() / ".claude",
        ]

        for mount_path in mount_locations:
            if not mount_path.exists():
                continue

            # Check for CLI credentials file (.credentials.json)
            creds_file = mount_path / ".credentials.json"
            if creds_file.exists():
                try:
                    with open(creds_file, "r") as f:
                        creds_data = json.load(f)

                    # CLI stores OAuth tokens
                    if creds_data.get("accessToken"):
                        logger.info(f"Found CLI credentials at {creds_file}")
                        return SessionCredentials(
                            credential_type="oauth",
                            oauth_token=creds_data.get("accessToken"),
                            refresh_token=creds_data.get("refreshToken"),
                            expires_at=creds_data.get("expiresAt"),
                        )
                except Exception as e:
                    logger.warning(f"Failed to load CLI credentials from {creds_file}: {e}")
                    continue

            # Check for API key in config
            config_file = mount_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config_data = json.load(f)

                    if config_data.get("anthropicApiKey"):
                        logger.info(f"Found API key in config at {config_file}")
                        return SessionCredentials(
                            credential_type="api_key",
                            api_key=config_data.get("anthropicApiKey"),
                        )
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_file}: {e}")
                    continue

        return None

    def _extract_credentials_from_settings(
        self,
        settings: Dict[str, Any]
    ) -> Optional[SessionCredentials]:
        """Extract credentials from a settings dictionary.

        Looks for credentials in common settings formats.
        """
        # Check for direct credentials object
        if "credentials" in settings:
            cred = settings["credentials"]
            return SessionCredentials(
                credential_type=cred.get("credential_type", "api_key"),
                api_key=cred.get("api_key"),
                oauth_token=cred.get("oauth_token") or cred.get("access_token"),
                refresh_token=cred.get("refresh_token"),
                expires_at=cred.get("expires_at"),
            )

        # Check for API key directly
        if "api_key" in settings or "anthropic_api_key" in settings:
            return SessionCredentials(
                credential_type="api_key",
                api_key=settings.get("api_key") or settings.get("anthropic_api_key"),
            )

        # Check for OAuth tokens
        if "oauth_token" in settings or "access_token" in settings:
            return SessionCredentials(
                credential_type="oauth",
                oauth_token=settings.get("oauth_token") or settings.get("access_token"),
                refresh_token=settings.get("refresh_token"),
                expires_at=settings.get("expires_at"),
            )

        return None

    def _check_dangerous_command(self, command: str, session_id: str) -> Optional[Tuple[str, str]]:
        """Check if a command matches dangerous patterns.

        Returns (category, pattern) if dangerous, None if safe.
        """
        for pattern, category in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return (category, pattern)
        return None

    def _log_blocked_command(
        self,
        session_id: str,
        session_name: str,
        command: str,
        category: str,
        pattern: str,
        tool_use_id: Optional[str]
    ) -> None:
        """Log blocked command to audit file."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "session_name": session_name,
            "command": command,
            "matched_pattern": pattern,
            "category": category,
            "tool_use_id": tool_use_id,
        }
        try:
            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    async def _pre_tool_validation_hook(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        tool_use_id: Optional[str],
        context: HookContext
    ) -> Tuple[bool, Optional[StreamMessage]]:
        """Validate tool inputs and block dangerous commands.

        Returns (allow, optional_message) - if allow is False, the command is blocked.
        """
        tool_name = context.tool_name if hasattr(context, 'tool_name') else None

        # Only validate Bash commands
        if tool_name != "Bash":
            return (True, None)

        command = input_data.get("command", "")
        if not command:
            return (True, None)

        # Check against dangerous patterns
        result = self._check_dangerous_command(command, session_id)
        if result:
            category, pattern = result
            session_info, _, _ = self._sessions.get(session_id, (None, None, None))
            session_name = session_info.name if session_info else "unknown"

            # Log to audit file
            self._log_blocked_command(
                session_id, session_name, command, category, pattern, tool_use_id
            )

            logger.warning(f"Session {session_id}: Blocked dangerous command ({category}): {command[:100]}...")

            # Return blocked message
            return (False, StreamMessage(
                type="blocked_command",
                content=f"Command blocked by security policy ({category})",
                tool_id=tool_use_id,
                tool_name="Bash",
                tool_input={"command": command},
                is_error=True,
                blocked_category=category,
                blocked_pattern=pattern,
            ))

        return (True, None)

    async def _ask_user_hook(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        tool_use_id: Optional[str],
        context: HookContext
    ) -> Dict[str, str]:
        """Handle AskUserQuestion tool - wait for user response.

        Returns the user's answers.
        """
        logger.info(f"Session {session_id}: AskUserQuestion triggered")

        # Parse questions from input_data
        tool_input = input_data.get("tool_input", {})
        raw_questions = tool_input.get("questions", [])

        questions = []
        for q in raw_questions:
            questions.append({
                "question": q.get("question", ""),
                "header": q.get("header", ""),
                "options": [
                    {"label": opt.get("label", ""), "description": opt.get("description", "")}
                    for opt in q.get("options", [])
                ],
                "multi_select": q.get("multiSelect", False),
            })

        # Store pending question
        question_data = QuestionData(
            question_id=tool_use_id or str(uuid.uuid4()),
            questions=questions,
            tool_use_id=tool_use_id or "",
        )
        self._pending_questions[session_id] = question_data

        # Update session status
        session_info, config, client = self._sessions.get(session_id, (None, None, None))
        if session_info:
            session_info.status = "waiting_for_input"

        # Wait for user response
        event = self._waiting_for_answer.get(session_id)
        if event:
            event.clear()
            logger.info(f"Session {session_id}: Waiting for user answer...")
            await event.wait()
            logger.info(f"Session {session_id}: Got user answer")

        # Get the response
        response_queue = self._question_responses.get(session_id)
        if response_queue:
            answers = await response_queue.get()
        else:
            answers = {}

        # Clean up
        self._pending_questions.pop(session_id, None)
        if session_info:
            session_info.status = "running"

        return answers

    async def submit_answer(
        self,
        session_id: str,
        question_id: str,
        answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Submit user's answers to a pending AskUserQuestion.

        Args:
            session_id: The session ID
            question_id: The question/tool_use_id from the question
            answers: Dict mapping question header to selected option label

        Returns:
            Dict with 'success' bool and optional error message
        """
        pending = self._pending_questions.get(session_id)
        if not pending:
            return {"success": False, "error": "No pending question"}

        if pending.question_id != question_id and pending.tool_use_id != question_id:
            return {"success": False, "error": "Question ID mismatch"}

        # Queue the answers and unblock the hook
        response_queue = self._question_responses.get(session_id)
        if response_queue:
            await response_queue.put(answers)

        event = self._waiting_for_answer.get(session_id)
        if event:
            event.set()

        return {"success": True}

    def get_pending_question(self, session_id: str) -> Optional[QuestionData]:
        """Get the current pending question for a session, if any."""
        return self._pending_questions.get(session_id)

    def has_pending_question(self, session_id: str) -> bool:
        """Check if a session has a pending question."""
        return session_id in self._pending_questions

    async def _pre_compact_hook(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        tool_use_id: Optional[str],
        context: HookContext
    ) -> StreamMessage:
        """Hook called before context compaction."""
        trigger = input_data.get("trigger", "unknown")
        logger.info(f"Session {session_id}: PreCompact triggered ({trigger})")

        return StreamMessage(
            type="status",
            status="compacting",
            content=f"Context compaction ({trigger}) - summarizing conversation to free space",
        )

    def _start_tool_execution(
        self,
        session_id: str,
        tool_use_id: str,
        tool_name: str,
        input_data: Dict[str, Any],
    ) -> None:
        """Record the start of a tool execution for timing."""
        execution = ToolExecution(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            input_data=input_data,
            start_time=time.time(),
        )
        self._tool_executions[tool_use_id] = execution
        logger.debug(f"Session {session_id}: Tool {tool_name} started (id: {tool_use_id})")

    def _complete_tool_execution(
        self,
        session_id: str,
        tool_use_id: str,
        result: Optional[str] = None,
        is_error: bool = False,
        was_blocked: bool = False,
    ) -> Optional[ToolExecution]:
        """Complete a tool execution and log it."""
        execution = self._tool_executions.pop(tool_use_id, None)
        if not execution:
            return None

        execution.end_time = time.time()
        execution.duration_ms = (execution.end_time - execution.start_time) * 1000
        execution.result = result[:500] if result and len(result) > 500 else result  # Truncate for log
        execution.is_error = is_error
        execution.was_blocked = was_blocked

        # Add to session history
        if session_id in self._session_tool_history:
            self._session_tool_history[session_id].append(execution)

        # Log to audit file
        self._log_tool_execution(session_id, execution)

        return execution

    def _log_tool_execution(self, session_id: str, execution: ToolExecution) -> None:
        """Log tool execution to audit file (PostToolUse logging)."""
        session_info, _, _ = self._sessions.get(session_id, (None, None, None))
        session_name = session_info.name if session_info else "unknown"

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "session_name": session_name,
            "tool_use_id": execution.tool_use_id,
            "tool_name": execution.tool_name,
            "duration_ms": round(execution.duration_ms, 2) if execution.duration_ms else 0,
            "is_error": execution.is_error,
            "was_blocked": execution.was_blocked,
            # Only log command for Bash, file_path for Read/Write/Edit
            "input_summary": self._summarize_tool_input(execution.tool_name, execution.input_data),
        }
        try:
            with open(self._tool_audit_log_path, "a") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write tool audit log: {e}")

        logger.info(
            f"[AUDIT] Session {session_name}: Tool {execution.tool_name} "
            f"completed in {execution.duration_ms:.1f}ms"
            f"{' (ERROR)' if execution.is_error else ''}"
            f"{' (BLOCKED)' if execution.was_blocked else ''}"
        )

    def _summarize_tool_input(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        """Create a safe summary of tool input for logging."""
        if tool_name == "Bash":
            cmd = input_data.get("command", "")
            # Truncate long commands
            return cmd[:200] + "..." if len(cmd) > 200 else cmd
        elif tool_name in ("Read", "Write", "Edit", "Glob"):
            return input_data.get("file_path", input_data.get("path", ""))[:200]
        elif tool_name == "Grep":
            return f"pattern: {input_data.get('pattern', '')[:100]}"
        elif tool_name == "WebFetch":
            return input_data.get("url", "")[:200]
        else:
            # Generic summary - just keys
            return ", ".join(input_data.keys())[:100]

    def _persist_output(self, session_id: str, msg: StreamMessage) -> None:
        """Persist a stream message to the session output buffer."""
        if session_id not in self._session_outputs:
            self._session_outputs[session_id] = []

        output = SessionOutput(
            timestamp=datetime.now().isoformat(),
            message_type=msg.type,
            content=msg.content[:5000] if msg.content and len(msg.content) > 5000 else msg.content,
            tool_name=msg.tool_name,
            tool_id=msg.tool_id,
            is_error=msg.is_error,
            metadata={
                k: v for k, v in {
                    "cost_usd": msg.cost_usd,
                    "input_tokens": msg.input_tokens,
                    "output_tokens": msg.output_tokens,
                    "status": msg.status,
                    "question_id": msg.question_id,
                    "blocked_category": msg.blocked_category,
                }.items() if v is not None
            } or None,
        )
        self._session_outputs[session_id].append(output)

    def save_session_outputs(self, session_id: str) -> Optional[Path]:
        """Save session outputs to a file for recovery."""
        outputs = self._session_outputs.get(session_id, [])
        if not outputs:
            return None

        session_info, config, _ = self._sessions.get(session_id, (None, None, None))
        if not session_info:
            return None

        output_file = self._output_dir / f"{session_id}.jsonl"
        try:
            with open(output_file, "w") as f:
                # Write header with session metadata
                header = {
                    "type": "header",
                    "session_id": session_id,
                    "session_name": session_info.name,
                    "project_path": session_info.project_path,
                    "model": session_info.model,
                    "created_at": session_info.created_at.isoformat(),
                    "saved_at": datetime.now().isoformat(),
                }
                f.write(json.dumps(header) + "\n")

                # Write each output
                for output in outputs:
                    entry = {
                        "type": "output",
                        "timestamp": output.timestamp,
                        "message_type": output.message_type,
                        "content": output.content,
                        "tool_name": output.tool_name,
                        "tool_id": output.tool_id,
                        "is_error": output.is_error,
                        "metadata": output.metadata,
                    }
                    f.write(json.dumps(entry) + "\n")

            logger.info(f"Session {session_id}: Saved {len(outputs)} outputs to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to save session outputs: {e}")
            return None

    def load_session_outputs(self, session_id: str) -> List[SessionOutput]:
        """Load previously saved session outputs."""
        output_file = self._output_dir / f"{session_id}.jsonl"
        if not output_file.exists():
            return []

        outputs = []
        try:
            with open(output_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "output":
                        outputs.append(SessionOutput(
                            timestamp=entry.get("timestamp", ""),
                            message_type=entry.get("message_type", ""),
                            content=entry.get("content"),
                            tool_name=entry.get("tool_name"),
                            tool_id=entry.get("tool_id"),
                            is_error=entry.get("is_error", False),
                            metadata=entry.get("metadata"),
                        ))
            logger.info(f"Loaded {len(outputs)} outputs for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to load session outputs: {e}")
        return outputs

    def get_tool_history(self, session_id: str) -> List[ToolExecution]:
        """Get the tool execution history for a session."""
        return self._session_tool_history.get(session_id, [])

    # ------------------------------------------------------------------
    # Fix(TAS-624): persistent SDK consumer
    # ------------------------------------------------------------------

    def _ensure_receive_loop(self, session_id: str, client: ClaudeSDKClient) -> asyncio.Queue:
        """Guarantee this session has a live consumer of the SDK stream.

        Returns the session's queue. Idempotent, and restarts the task if a
        previous one died -- a session whose consumer is gone is exactly the
        state that strands the SDK's buffer, so "missing" must be recoverable
        rather than fatal.
        """
        queue = self._turn_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._turn_queues[session_id] = queue

        task = self._receive_tasks.get(session_id)
        if task is None or task.done():
            if task is not None and task.done():
                exc = task.exception() if not task.cancelled() else None
                logger.warning(
                    f"Session {session_id}: receive loop had stopped "
                    f"({exc!r}); restarting so the SDK stream keeps a consumer"
                )
            self._receive_tasks[session_id] = asyncio.create_task(
                self._receive_loop(session_id, client)
            )
        return queue

    def _offer(self, session_id: str, queue: asyncio.Queue, item: Any) -> None:
        """Put an item on a session's queue WITHOUT ever blocking.

        The no-blocking property is the whole point of this fix, so this method
        must never await. Past the retention cap, ordinary messages are dropped
        (they remain in _session_outputs) while turn boundaries always go on --
        dropping one would leave the next RPC waiting for an end that never
        comes.
        """
        if isinstance(item, _TurnEnd):
            queue.put_nowait(item)
            return

        if queue.qsize() >= _MAX_UNDRAINED_MESSAGES:
            dropped = self._dropped_messages.get(session_id, 0) + 1
            self._dropped_messages[session_id] = dropped
            if dropped == 1:
                logger.warning(
                    f"Session {session_id}: over {_MAX_UNDRAINED_MESSAGES} undrained "
                    f"messages -- nobody is reading this turn. Dropping from the live "
                    f"queue only; output is still recorded in session outputs."
                )
            return

        queue.put_nowait(item)

    async def _receive_loop(self, session_id: str, client: ClaudeSDKClient) -> None:
        """Consume the SDK message stream for the LIFE OF THE CLIENT.

        Fix(TAS-624). send_prompt used to iterate ``client.receive_messages()``
        itself, with no break on ResultMessage. Four things followed from that,
        and all four are removed by moving the consumer out here:

        1. **The RPC never completed.** ``receive_messages()`` ends only when the
           CLI transport closes, and a ``ClaudeSDKClient`` keeps its CLI alive
           across turns by design -- so the generator parked forever at the
           ``async for`` and the two statements after it (status = "idle",
           save_session_outputs) were unreachable in normal operation.
        2. **The session never went idle.** Tier2Session.is_idle() reads a status
           that only got set after that loop exited, so a T2/T3 session reported
           RUNNING forever after its first prompt.
        3. **Turn 2 onward split its output.** Each prompt began a NEW consumer of
           the SAME anyio memory object stream, which delivers each message to
           exactly one receiver. The turn-1 RPC was still pulling, so roughly half
           of every later turn went to the abandoned iterator.
        4. **The control plane could die.** An async generator only advances when
           its consumer pulls. Whenever it was suspended at a ``yield`` -- gRPC
           flow control, a slow or cancelled client, or the question path that
           yields and then awaits an answer -- nothing drained the SDK's 100-slot
           buffer. Once full, the SDK's reader task parks, and interrupt(),
           set_model() and every hook callback stop working with no error raised.

        A ResultMessage is therefore a **turn boundary, not a stop condition**:
        end-of-turn work runs and the loop keeps consuming.
        """
        queue = self._turn_queues[session_id]
        try:
            async for message in client.receive_messages():
                entry = self._sessions.get(session_id)
                if entry is None:
                    # Session deleted underneath us; nothing left to serve.
                    logger.info(f"Session {session_id}: gone, ending receive loop")
                    return
                session_info = entry[0]

                # Isolate per-message failures. A conversion or persistence bug is
                # OURS, and letting one abort the loop strands the SDK buffer --
                # the precise failure this loop exists to prevent.
                try:
                    converted = self._convert_message(message, session_info)
                except Exception as exc:
                    logger.error(
                        f"Session {session_id}: failed to convert "
                        f"{type(message).__name__}: {exc}. Continuing to drain."
                    )
                    converted = []

                for stream_msg in converted:
                    try:
                        self._track_and_persist(session_id, stream_msg)
                    except Exception as exc:
                        logger.error(
                            f"Session {session_id}: failed to record "
                            f"{stream_msg.type}: {exc}. Continuing to drain."
                        )
                    self._offer(session_id, queue, stream_msg)

                if isinstance(message, ResultMessage):
                    session_info.status = "idle"
                    dropped = self._dropped_messages.pop(session_id, 0)
                    if dropped:
                        logger.warning(
                            f"Session {session_id}: turn ended having dropped {dropped} "
                            f"message(s) from the live queue (still in session outputs)"
                        )
                    try:
                        self.save_session_outputs(session_id)
                    except Exception as exc:
                        logger.error(f"Session {session_id}: save outputs failed - {exc}")
                    logger.info(f"Session {session_id}: prompt completed")
                    seq = self._result_seq.get(session_id, 0) + 1
                    self._result_seq[session_id] = seq
                    self._offer(session_id, queue, _TurnEnd(seq))

            # Falling out means the SDK CLOSED the stream: the CLI process is gone.
            logger.info(f"Session {session_id}: SDK stream closed, receive loop ending")
            self._finish_turn_with_error(session_id, queue, "Claude CLI transport closed")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Session {session_id}: receive loop error - {e}")
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry[0].status = "error"
            self._finish_turn_with_error(session_id, queue, str(e))

    def _finish_turn_with_error(
        self, session_id: str, queue: asyncio.Queue, detail: str
    ) -> None:
        """Release any RPC waiting on this session, with the reason.

        Without this an RPC in progress when the transport dies would wait on a
        turn boundary that can no longer arrive -- trading the old never-ending
        turn for a new one.
        """
        self._offer(session_id, queue, StreamMessage(type="error", content=detail, is_error=True))
        # Unconditional: the transport is gone, so no sequenced marker can ever
        # arrive for whichever turn the waiter is serving.
        self._offer(session_id, queue, _TurnEnd(None))

    def _track_and_persist(self, session_id: str, stream_msg: StreamMessage) -> None:
        """Record a converted message: tool audit trail, then output buffer.

        Fix(TAS-624): this runs in the receive loop, not in the RPC, so the audit
        trail and the recovery buffer stay complete even when no client is
        reading -- a cancelled RPC used to take the tool history with it.
        """
        if stream_msg.type == "tool_use" and stream_msg.tool_id:
            self._start_tool_execution(
                session_id,
                stream_msg.tool_id,
                stream_msg.tool_name or "unknown",
                stream_msg.tool_input or {},
            )
        elif stream_msg.type == "tool_result" and stream_msg.tool_id:
            self._complete_tool_execution(
                session_id,
                stream_msg.tool_id,
                result=stream_msg.content,
                is_error=stream_msg.is_error,
            )
        elif stream_msg.type == "blocked_command" and stream_msg.tool_id:
            self._complete_tool_execution(
                session_id,
                stream_msg.tool_id,
                result=stream_msg.content,
                is_error=True,
                was_blocked=True,
            )

        self._persist_output(session_id, stream_msg)

    @staticmethod
    def _drain_stale(queue: asyncio.Queue) -> int:
        """Empty a queue before a new turn begins.

        Anything still on it belongs to a PREVIOUS turn whose reader went away.
        Replaying it as part of the new turn would attribute one prompt's output
        to another; it is already in the session's output buffer either way.
        """
        dropped = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            dropped += 1

    async def send_prompt(
        self, session_id: str, prompt: str
    ) -> AsyncIterator[StreamMessage]:
        """Send a prompt and stream responses."""
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session_info, config, client = self._sessions[session_id]
        session_info.status = "running"
        logger.info(f"Session {session_id}: sending prompt ({len(prompt)} chars)")

        try:
            # Build hooks
            hooks = {}

            if config.enable_dangerous_command_blocking:
                # Create validation hook
                async def pre_tool_hook(input_data, tool_use_id, context):
                    allow, msg = await self._pre_tool_validation_hook(
                        session_id, input_data, tool_use_id, context
                    )
                    if not allow:
                        # Yield the blocked message through the stream
                        return {"decision": "block", "reason": msg.content if msg else "Blocked"}
                    return {}

                hooks["PreToolUse"] = [HookMatcher(matcher="Bash", hooks=[pre_tool_hook])]

            # Build options for the SDK
            options = ClaudeAgentOptions(
                model=config.model,
                cwd=config.project_path,
                permission_mode=config.permission_mode,
                # ClaudeAgentOptions.allowed_tools is declared `list[str]`
                # with default_factory=list, so None was never a legal value --
                # it merely used to be harmless, because every consumer of it
                # was behind a truthiness check. The Skills feature added an
                # unguarded one (`list(self._options.allowed_tools)` in
                # subprocess_cli._apply_skills_defaults, reached from
                # _build_command on EVERY connect), so from claude-agent-sdk
                # 0.2.x an empty allow-list raises
                # `TypeError: 'NoneType' object is not iterable` before the CLI
                # is even spawned -- i.e. every Tier 2/Tier 3 session that does
                # not name an explicit allow-list, which is the common case.
                # Pass the empty list the type always asked for.
                allowed_tools=config.allowed_tools or [],
                disallowed_tools=config.disallowed_tools if config.disallowed_tools else ["EnterPlanMode", "ExitPlanMode"],
                max_turns=config.max_turns,
                continue_conversation=config.continue_conversation,
                hooks=hooks if hooks else None,
            )

            # Create client if not exists
            if client is None:
                client = ClaudeSDKClient(options)
                await client.connect()
                self._sessions[session_id] = (session_info, config, client)

            # Fix(TAS-624): the SDK stream is consumed by a persistent background
            # loop, not by this generator. This RPC only drains the queue that
            # loop fills, for the length of ONE turn. See _receive_loop for the
            # four defects that consuming it here caused.
            queue = self._ensure_receive_loop(session_id, client)

            # Anything already queued belongs to a previous turn whose reader
            # went away -- it is in the output buffer, and replaying it here
            # would bill one prompt's output to another.
            stale = self._drain_stale(queue)
            if stale:
                logger.info(
                    f"Session {session_id}: discarded {stale} message(s) left "
                    f"over from an earlier turn before sending a new prompt"
                )

            # ClaudeSDKClient has no `send`. The method that writes a prompt
            # into a streaming session is `query(prompt, session_id="default")`,
            # and generate_summary in this same file already calls it -- this one
            # call site was the outlier. It raised
            # `AttributeError: 'ClaudeSDKClient' object has no attribute 'send'`
            # on the FIRST turn of every Tier 2/Tier 3 session, so no prompt has
            # ever reached Claude through this path.
            #
            # It stayed invisible because the test double defines send(): a stub
            # is free to implement an interface the real class does not have, and
            # a suite built on one then proves only that the stub is
            # self-consistent. test_stub_matches_sdk_interface.py now pins the
            # double to the real class so this cannot recur.
            # Fix(TAS-808): claim this turn's number BEFORE the prompt goes in,
            # so the marker the CLI will eventually emit for it is already
            # matchable. Claiming after would leave a window where a result
            # arriving fast is tagged higher than the turn that caused it.
            my_seq = self._prompt_seq.get(session_id, 0) + 1
            self._prompt_seq[session_id] = my_seq

            await client.query(prompt)

            while True:
                item = await queue.get()
                if isinstance(item, _TurnEnd):
                    if item.seq is None or item.seq >= my_seq:
                        return
                    # Older than mine: an earlier turn (usually one that was
                    # interrupted) has only just closed, and its own reader is
                    # gone. Before this check the FIRST such marker ended THIS
                    # turn instead, returning with no output at all.
                    logger.info(
                        f"Session {session_id}: dropped a turn marker for turn "
                        f"{item.seq} while serving turn {my_seq}"
                    )
                    continue

                stream_msg = item
                if stream_msg.type == "question":
                    # Awaiting the answer suspends THIS generator, which is now
                    # safe: the receive loop goes on draining the SDK meanwhile.
                    # Before the fix, this await was the most reliable way to
                    # fill the SDK's buffer and kill the control plane.
                    yield stream_msg
                    await self._ask_user_hook(
                        session_id,
                        {"tool_input": {"questions": stream_msg.question_options}},
                        stream_msg.question_id,
                        None
                    )
                    # The hook sends the answer back to the SDK.
                else:
                    yield stream_msg

        except Exception as e:
            session_info.status = "error"
            # logger.exception, not logger.error: the message alone stripped the
            # traceback, and the defect this replaced was a TypeError raised
            # inside the SDK's own command builder. Without the stack the only
            # evidence anywhere was the bare str(e) echoed to the client.
            logger.exception(f"Session {session_id}: error - {e}")
            yield StreamMessage(
                type="error",
                content=str(e),
                is_error=True,
            )

    def _convert_message(self, message: Any, session_info: SessionInfo) -> List[StreamMessage]:
        """Convert SDK message to StreamMessage(s)."""
        messages = []

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    messages.append(StreamMessage(
                        type="text",
                        content=block.text,
                    ))
                elif isinstance(block, ThinkingBlock):
                    messages.append(StreamMessage(
                        type="thinking",
                        content=block.thinking,
                    ))
                elif isinstance(block, ToolUseBlock):
                    # Check for AskUserQuestion
                    if block.name == "AskUserQuestion":
                        tool_input = block.input if isinstance(block.input, dict) else {}
                        questions = tool_input.get("questions", [])
                        messages.append(StreamMessage(
                            type="question",
                            question_id=block.id,
                            question_text=questions[0].get("question", "") if questions else "",
                            question_options=questions,
                            allow_custom=True,
                        ))
                    else:
                        messages.append(StreamMessage(
                            type="tool_use",
                            tool_id=block.id,
                            tool_name=block.name,
                            tool_input=block.input if isinstance(block.input, dict) else {},
                        ))
                elif isinstance(block, ToolResultBlock):
                    content = block.content
                    if isinstance(content, list):
                        content = "\n".join(str(c) for c in content)
                    messages.append(StreamMessage(
                        type="tool_result",
                        tool_id=block.tool_use_id,
                        content=str(content),
                        is_error=getattr(block, 'is_error', False),
                    ))

        elif isinstance(message, ResultMessage):
            # Capture SDK session ID
            if hasattr(message, 'session_id') and message.session_id:
                session_info.sdk_session_id = message.session_id

            messages.append(StreamMessage(
                type="cost",
                cost_usd=getattr(message, 'total_cost_usd', None),
                input_tokens=getattr(message, 'input_tokens', None),
                output_tokens=getattr(message, 'output_tokens', None),
            ))
            messages.append(StreamMessage(
                type="status",
                status="completed",
            ))

        elif isinstance(message, SystemMessage):
            messages.append(StreamMessage(
                type="status",
                status=getattr(message, 'subtype', 'system'),
            ))

        return messages

    async def interrupt(self, session_id: str) -> bool:
        """Interrupt current execution."""
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session_info, config, client = self._sessions[session_id]

        if client and hasattr(client, 'interrupt'):
            try:
                await client.interrupt()
                session_info.status = "idle"
                logger.info(f"Session {session_id}: interrupted")
                return True
            except Exception as e:
                logger.error(f"Session {session_id}: interrupt failed - {e}")
                return False
        else:
            session_info.status = "idle"
            logger.info(f"Session {session_id}: interrupt requested (status update only)")
            return True

    async def delete_session(self, session_id: str) -> None:
        """Delete/close a session."""
        if session_id in self._sessions:
            session_info, config, client = self._sessions.pop(session_id)

            # Fix(TAS-624): stop this session's persistent SDK consumer BEFORE
            # disconnecting, then release any RPC still waiting on a turn
            # boundary that will now never arrive.
            task = self._receive_tasks.pop(session_id, None)
            if task is not None and not task.done():
                task.cancel()
            queue = self._turn_queues.pop(session_id, None)
            if queue is not None:
                # Unbounded, so put_nowait cannot raise QueueFull here.
                # Unconditional: the session is going away, release everyone.
                queue.put_nowait(_TurnEnd(None))
            self._dropped_messages.pop(session_id, None)
            self._prompt_seq.pop(session_id, None)
            self._result_seq.pop(session_id, None)

            # Disconnect client if connected
            if client:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting client: {e}")

            # Clean up ephemeral credentials (restore original env values)
            if session_info.has_ephemeral_credentials:
                self._cleanup_ephemeral_credentials(session_id)

            # Clean up question handling
            self._pending_questions.pop(session_id, None)
            self._question_responses.pop(session_id, None)
            self._waiting_for_answer.pop(session_id, None)

            # Clean up tool tracking
            self._session_tool_history.pop(session_id, None)
            # Clean up any pending tool executions for this session
            pending_tool_ids = [
                tid for tid, exec in self._tool_executions.items()
                if self._is_tool_from_session(tid, session_id)
            ]
            for tid in pending_tool_ids:
                self._tool_executions.pop(tid, None)

            # Clean up output persistence (keep the file for recovery)
            self._session_outputs.pop(session_id, None)

            logger.info(f"Session {session_id}: deleted")

    def _is_tool_from_session(self, tool_use_id: str, session_id: str) -> bool:
        """Check if a tool_use_id belongs to a session (simple check, can be enhanced)."""
        # For now, we rely on the fact that tool_use_ids are unique and scoped
        # A more robust implementation would track session_id in ToolExecution
        return True  # Conservative - clean up all if we can't determine

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session info."""
        if session_id in self._sessions:
            return self._sessions[session_id][0]
        return None

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions."""
        return [info for info, _, _ in self._sessions.values()]

    def get_active_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_uptime(self) -> str:
        """Get server uptime as a human-readable string."""
        delta = datetime.now(timezone.utc) - self._startup_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    # --- Session State Persistence ---

    def save_session_state(self, session_id: str) -> Optional[Path]:
        """Save session state to disk for recovery after server restart."""
        if session_id not in self._sessions:
            return None

        session_info, config, _ = self._sessions[session_id]
        state_dir = Path.home() / ".claude" / "state" / "orchestration" / "grpc_sessions"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / f"{session_id}.json"

        state = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "session": {
                "id": session_info.id,
                "name": session_info.name,
                "status": session_info.status,
                "project_path": session_info.project_path,
                "model": session_info.model,
                "created_at": session_info.created_at.isoformat(),
                "sdk_session_id": session_info.sdk_session_id,
                "context_tokens": session_info.context_tokens,
                "max_context_tokens": session_info.max_context_tokens,
                "continuation_count": session_info.continuation_count,
            },
            "config": {
                "project_path": config.project_path,
                "model": config.model,
                "permission_mode": config.permission_mode,
                "allowed_tools": config.allowed_tools,
                "disallowed_tools": config.disallowed_tools,
                "max_turns": config.max_turns,
                "continue_conversation": config.continue_conversation,
                "environment": config.environment,
                "enable_dangerous_command_blocking": config.enable_dangerous_command_blocking,
                "headless": config.headless,
            },
            "tool_history": [
                {
                    "tool_use_id": ex.tool_use_id,
                    "tool_name": ex.tool_name,
                    "duration_ms": ex.duration_ms,
                    "is_error": ex.is_error,
                    "was_blocked": ex.was_blocked,
                }
                for ex in self._session_tool_history.get(session_id, [])[-100:]  # Last 100
            ],
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Session {session_id}: State saved to {state_file}")
            return state_file
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return None

    def save_all_session_states(self) -> int:
        """Save all session states (called on graceful shutdown)."""
        count = 0
        for session_id in self._sessions:
            if self.save_session_state(session_id):
                count += 1
        logger.info(f"Saved {count} session states")
        return count

    async def restore_session(self, session_id: str) -> Optional[SessionInfo]:
        """Restore a session from saved state."""
        state_dir = Path.home() / ".claude" / "state" / "orchestration" / "grpc_sessions"
        state_file = state_dir / f"{session_id}.json"

        if not state_file.exists():
            logger.warning(f"Session state file not found: {state_file}")
            return None

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            # Recreate session from state
            session_data = state["session"]
            config_data = state["config"]

            config = SessionConfig(
                project_path=config_data["project_path"],
                model=config_data.get("model", "sonnet"),
                permission_mode=config_data.get("permission_mode", "default"),
                allowed_tools=config_data.get("allowed_tools", []),
                disallowed_tools=config_data.get("disallowed_tools", []),
                max_turns=config_data.get("max_turns"),
                continue_conversation=config_data.get("continue_conversation", False),
                environment=config_data.get("environment", {}),
                enable_dangerous_command_blocking=config_data.get("enable_dangerous_command_blocking", True),
                headless=config_data.get("headless", True),
            )

            session_info = SessionInfo(
                id=session_data["id"],
                name=session_data["name"],
                status="idle",  # Reset status on restore
                project_path=session_data["project_path"],
                model=session_data["model"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                sdk_session_id=session_data.get("sdk_session_id"),
                context_tokens=session_data.get("context_tokens", 0),
                max_context_tokens=session_data.get("max_context_tokens", 200000),
                continuation_count=session_data.get("continuation_count", 0),
            )

            # Initialize tracking structures
            self._question_responses[session_id] = asyncio.Queue()
            self._waiting_for_answer[session_id] = asyncio.Event()
            self._waiting_for_answer[session_id].set()
            self._session_tool_history[session_id] = []
            self._session_outputs[session_id] = []

            # Restore outputs if available
            saved_outputs = self.load_session_outputs(session_id)
            if saved_outputs:
                self._session_outputs[session_id] = saved_outputs

            # Store session (client will be created on first prompt)
            self._sessions[session_id] = (session_info, config, None)

            logger.info(f"Session {session_id}: Restored from state file")
            return session_info

        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}")
            return None

    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """List all saved session states that can be restored."""
        state_dir = Path.home() / ".claude" / "state" / "orchestration" / "grpc_sessions"
        if not state_dir.exists():
            return []

        sessions = []
        for state_file in state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                session = state.get("session", {})
                sessions.append({
                    "session_id": session.get("id"),
                    "session_name": session.get("name"),
                    "project_path": session.get("project_path"),
                    "model": session.get("model"),
                    "created_at": session.get("created_at"),
                    "saved_at": state.get("saved_at"),
                    "is_active": session.get("id") in self._sessions,
                })
            except Exception as e:
                logger.warning(f"Failed to read state file {state_file}: {e}")
                continue

        return sorted(sessions, key=lambda s: s.get("saved_at", ""), reverse=True)

    # --- Account/OAuth Management ---

    def _load_accounts(self) -> None:
        """Load saved account credentials from disk."""
        if not self._accounts_file.exists():
            return
        try:
            with open(self._accounts_file, "r") as f:
                data = json.load(f)
            for account_id, cred_data in data.items():
                self._accounts[account_id] = AccountCredentials(
                    account_id=account_id,
                    credential_type=cred_data.get("credential_type", "api_key"),
                    api_key=cred_data.get("api_key"),
                    access_token=cred_data.get("access_token"),
                    refresh_token=cred_data.get("refresh_token"),
                    expires_at=cred_data.get("expires_at"),
                    created_at=cred_data.get("created_at"),
                    updated_at=cred_data.get("updated_at"),
                )
            logger.info(f"Loaded {len(self._accounts)} account credentials")
        except Exception as e:
            logger.error(f"Failed to load accounts: {e}")

    def _save_accounts(self) -> None:
        """Save account credentials to disk."""
        try:
            self._accounts_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for account_id, cred in self._accounts.items():
                data[account_id] = {
                    "credential_type": cred.credential_type,
                    "api_key": cred.api_key,
                    "access_token": cred.access_token,
                    "refresh_token": cred.refresh_token,
                    "expires_at": cred.expires_at,
                    "created_at": cred.created_at,
                    "updated_at": cred.updated_at,
                }
            with open(self._accounts_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} account credentials")
        except Exception as e:
            logger.error(f"Failed to save accounts: {e}")

    async def _setup_account_env(self, account_id: str) -> bool:
        """Set up environment variables for an account's credentials.

        For OAuth accounts, checks if token needs refresh and refreshes if needed.
        """
        cred = self._accounts.get(account_id)
        if not cred:
            logger.warning(f"Account {account_id} not found")
            return False

        if cred.credential_type == "oauth":
            # Check if token needs refresh
            if cred.expires_at:
                now = int(time.time())
                # Refresh if expires within 5 minutes
                if cred.expires_at - now < 300:
                    logger.info(f"Account {account_id}: Token expiring soon, attempting refresh")
                    refreshed = await self.refresh_token(account_id)
                    if not refreshed.get("success"):
                        logger.error(f"Account {account_id}: Token refresh failed")
                        return False
                    # Reload credentials after refresh
                    cred = self._accounts.get(account_id)

            if cred and cred.access_token:
                import os
                os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = cred.access_token
                logger.info(f"Account {account_id}: Set OAuth access token")
                return True

        elif cred.credential_type == "api_key" and cred.api_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = cred.api_key
            logger.info(f"Account {account_id}: Set API key")
            return True

        return False

    def set_account_credentials(
        self,
        account_id: str,
        credential_type: str,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        expires_at: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Set credentials for an account."""
        now = datetime.now().isoformat()

        existing = self._accounts.get(account_id)
        created_at = existing.created_at if existing else now

        self._accounts[account_id] = AccountCredentials(
            account_id=account_id,
            credential_type=credential_type,
            api_key=api_key,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            created_at=created_at,
            updated_at=now,
        )
        self._save_accounts()
        logger.info(f"Account {account_id}: Credentials set ({credential_type})")
        return {"success": True}

    def _find_cli(self) -> Optional[str]:
        """Find Claude Code CLI binary."""
        # Try common locations
        if cli := shutil.which("claude"):
            return cli

        locations = [
            Path.home() / ".npm-global/bin/claude",
            Path("/usr/local/bin/claude"),
            Path.home() / ".local/bin/claude",
            Path.home() / "node_modules/.bin/claude",
            Path.home() / ".yarn/bin/claude",
            Path.home() / ".claude/local/claude",
        ]

        for path in locations:
            if path.exists() and path.is_file():
                return str(path)

        logger.warning("Claude Code CLI not found - OAuth operations will not work")
        return None

    def get_auth_status(self) -> Dict[str, Any]:
        """Check Claude Code CLI authentication status.

        Uses 'claude auth status' to check if the CLI is authenticated.
        """
        if not self._cli_path:
            return {
                "authenticated": False,
                "error": "Claude Code CLI not found",
            }

        try:
            result = subprocess.run(
                [self._cli_path, "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                try:
                    status = json.loads(result.stdout)
                    return {
                        "authenticated": status.get("authenticated", False),
                        "account_type": status.get("accountType"),
                        "email": status.get("email"),
                    }
                except json.JSONDecodeError:
                    # Fallback: if stdout contains success message
                    if "authenticated" in result.stdout.lower():
                        return {"authenticated": True}
            else:
                return {
                    "authenticated": False,
                    "error": result.stderr or "Not authenticated",
                }

        except subprocess.TimeoutExpired:
            return {"authenticated": False, "error": "Auth status check timed out"}
        except Exception as e:
            return {"authenticated": False, "error": str(e)}

    def start_oauth(self, redirect_uri: str) -> Dict[str, Any]:
        """Start OAuth login flow using Claude Code CLI.

        This initiates the CLI's OAuth flow. For headless/server environments,
        use 'claude auth login' or 'claude auth setup-token' instead.

        Note: Claude Code OAuth tokens are restricted to Claude Code usage only
        and cannot be used for direct Anthropic API calls.

        Args:
            redirect_uri: Not used - CLI manages redirect internally

        Returns:
            Dict with instructions for completing OAuth
        """
        if not self._cli_path:
            return {
                "success": False,
                "error": "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            }

        # Check current auth status first
        status = self.get_auth_status()
        if status.get("authenticated"):
            return {
                "success": True,
                "message": "Already authenticated",
                "email": status.get("email"),
            }

        # For server environments, provide instructions
        # The actual OAuth flow requires user interaction via CLI
        return {
            "success": False,
            "requires_cli_login": True,
            "instructions": (
                "OAuth authentication requires user interaction with the Claude Code CLI.\n"
                "Run one of the following commands:\n"
                "  - Interactive: claude auth login\n"
                "  - Headless/CI: claude auth setup-token (generate token from web)\n"
                "See: https://code.claude.com/docs/en/iam"
            ),
        }

    async def complete_oauth(self, code: str, state: str) -> Dict[str, Any]:
        """Complete OAuth flow - not applicable for CLI-based auth.

        The Claude Code CLI handles the OAuth flow internally.
        This method is kept for API compatibility but returns an error
        directing users to use CLI-based authentication.

        Args:
            code: Not used - CLI handles token exchange
            state: Not used - CLI handles state validation

        Returns:
            Dict with error message explaining CLI-based auth
        """
        return {
            "success": False,
            "error": (
                "OAuth token exchange is handled by the Claude Code CLI. "
                "Use 'claude auth login' to authenticate, or 'claude auth setup-token' "
                "for headless environments. The gRPC server will use the CLI's credentials."
            ),
        }

    def get_account_status(self, account_id: str) -> Dict[str, Any]:
        """Get status of an account's credentials."""
        cred = self._accounts.get(account_id)
        if not cred:
            return {
                "exists": False,
                "credential_type": "",
                "is_valid": False,
            }

        is_valid = False
        if cred.credential_type == "api_key":
            is_valid = bool(cred.api_key)
        elif cred.credential_type == "oauth":
            if cred.access_token:
                # Check if token is expired
                if cred.expires_at:
                    is_valid = cred.expires_at > int(time.time())
                else:
                    is_valid = True

        return {
            "exists": True,
            "credential_type": cred.credential_type,
            "is_valid": is_valid,
            "token_expires_at": cred.expires_at,
        }

    async def refresh_token(self, account_id: str) -> Dict[str, Any]:
        """Refresh OAuth token for an account.

        For CLI-based OAuth, token refresh is handled automatically by the
        Claude Code CLI. This method checks the current auth status and
        provides guidance if re-authentication is needed.

        For API key accounts, no refresh is needed.
        """
        cred = self._accounts.get(account_id)
        if not cred:
            return {"success": False, "error": "Account not found"}

        if cred.credential_type == "api_key":
            return {
                "success": True,
                "message": "API key credentials do not require refresh",
            }

        if cred.credential_type != "oauth":
            return {"success": False, "error": "Unknown credential type"}

        # For OAuth, check CLI auth status - the CLI manages its own token refresh
        status = self.get_auth_status()

        if status.get("authenticated"):
            logger.info(f"Account {account_id}: CLI authentication is valid")
            return {
                "success": True,
                "message": "CLI authentication is valid",
                "email": status.get("email"),
            }
        else:
            logger.warning(f"Account {account_id}: CLI authentication needs renewal")
            return {
                "success": False,
                "requires_reauth": True,
                "error": "CLI authentication has expired or is invalid",
                "instructions": (
                    "Re-authenticate using the Claude Code CLI:\n"
                    "  - Interactive: claude auth login\n"
                    "  - Headless/CI: claude auth setup-token\n"
                ),
            }

    # --- Session Summary Generation ---

    async def generate_summary(self, session_id: str, max_chars: int = 150) -> Dict[str, Any]:
        """Generate an AI summary of session activity using Claude.

        Uses the session's output history to create a concise summary.
        """
        if session_id not in self._sessions:
            return {"success": False, "error": "Session not found"}

        session_info, config, client = self._sessions[session_id]

        # Get session output
        outputs = self._session_outputs.get(session_id, [])
        if not outputs:
            return {"success": False, "error": "No session output to summarize"}

        # Build output text from saved outputs
        output_text = ""
        for out in outputs[-50:]:  # Last 50 messages
            if out.content:
                output_text += f"[{out.message_type}] {out.content[:500]}\n"

        if len(output_text) < 100:
            return {"success": False, "error": "Not enough output to summarize"}

        # Truncate to ~8K chars for context
        recent_output = output_text[-8000:] if len(output_text) > 8000 else output_text

        # Create summary prompt
        prompt = f"""Analyze this Claude Code session output and provide a VERY brief summary (1-2 sentences, max {max_chars} chars) of what was accomplished. Focus on the main actions/changes made.

Session output (truncated):
{recent_output}

Respond with ONLY the summary, no preamble or explanation. Examples of good summaries:
- "Added dark mode toggle to settings page with theme persistence"
- "Fixed authentication bug in login flow, updated tests"
- "Refactored API client, added error handling for network failures"
"""

        try:
            # Use a separate Claude call for summary generation
            summary_client = ClaudeSDKClient(ClaudeAgentOptions(
                model="haiku",  # Use haiku for fast, cheap summary
                cwd=config.project_path,
                permission_mode="default",
                max_turns=1,
            ))
            await summary_client.connect()

            # Query for summary
            result = await summary_client.query(prompt)
            await summary_client.disconnect()

            # Extract text from result
            summary = ""
            if hasattr(result, 'content'):
                for block in result.content:
                    if hasattr(block, 'text'):
                        summary = block.text.strip()
                        break

            if summary:
                # Truncate if needed
                if len(summary) > max_chars:
                    summary = summary[:max_chars-3] + "..."

                # Update session info
                session_info.summary = summary
                logger.info(f"Session {session_id}: Generated summary: {summary[:50]}...")

                return {"success": True, "summary": summary}
            else:
                return {"success": False, "error": "No summary generated"}

        except Exception as e:
            logger.error(f"Session {session_id}: Summary generation failed: {e}")
            return {"success": False, "error": str(e)}

    # --- Subagent (Task tool) Tracking ---

    def _start_subagent(self, session_id: str, agent_id: str, prompt: Optional[str] = None) -> None:
        """Record the start of a subagent execution."""
        self._subagents[agent_id] = SubagentInfo(
            agent_id=agent_id,
            session_id=session_id,
            start_time=time.time(),
            prompt=prompt,
            status="running",
        )
        logger.info(f"Session {session_id}: Subagent {agent_id[:8]}... started")

    def _complete_subagent(
        self,
        agent_id: str,
        result: Optional[str] = None,
        is_error: bool = False
    ) -> Optional[StreamMessage]:
        """Complete a subagent execution and return a SubagentComplete message."""
        subagent = self._subagents.pop(agent_id, None)
        if not subagent:
            return None

        duration_ms = (time.time() - subagent.start_time) * 1000
        subagent.status = "error" if is_error else "completed"

        logger.info(
            f"Session {subagent.session_id}: Subagent {agent_id[:8]}... completed "
            f"in {duration_ms:.1f}ms{' (ERROR)' if is_error else ''}"
        )

        return StreamMessage(
            type="subagent_complete",
            agent_id=agent_id,
            content=result[:1000] if result and len(result) > 1000 else result,
            agent_duration_ms=duration_ms,
            is_error=is_error,
        )

    async def _subagent_stop_hook(
        self,
        session_id: str,
        input_data: Dict[str, Any],
        tool_use_id: Optional[str],
        context: HookContext
    ) -> Optional[StreamMessage]:
        """Hook called when a subagent (Task tool) stops.

        This hook is triggered when a spawned agent completes its work.
        """
        agent_id = input_data.get("agent_id") or tool_use_id or str(uuid.uuid4())
        result = input_data.get("result", "")
        is_error = input_data.get("is_error", False)

        return self._complete_subagent(agent_id, result, is_error)
