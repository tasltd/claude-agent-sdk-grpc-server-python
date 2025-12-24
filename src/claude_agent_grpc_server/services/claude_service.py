"""
gRPC Service Implementation for Claude Agent.

This module implements the ClaudeAgentService gRPC service,
providing session management and streaming chat capabilities.
"""

import grpc
from typing import AsyncIterator
from datetime import datetime, timezone
import json
import logging

from ..proto import claude_agent_pb2 as pb2
from ..proto import claude_agent_pb2_grpc as pb2_grpc
from ..sdk.session_manager import SessionManager, SessionConfig, StreamMessage

logger = logging.getLogger(__name__)

# Version of the gRPC server
SERVER_VERSION = "0.1.0"


class ClaudeAgentServicer(pb2_grpc.ClaudeAgentServiceServicer):
    """
    gRPC service implementation for Claude Agent.

    This servicer handles all gRPC requests for Claude sessions,
    including creation, chat streaming, and lifecycle management.
    """

    def __init__(self, session_manager: SessionManager | None = None):
        """
        Initialize the servicer.

        Args:
            session_manager: Optional SessionManager instance. If not provided,
                           a new one will be created.
        """
        self.session_manager = session_manager or SessionManager()
        logger.info("ClaudeAgentServicer initialized")

    async def CreateSession(
        self, request: pb2.CreateSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb2.Session:
        """Create a new Claude session."""
        logger.info(f"CreateSession: project_path={request.config.project_path}")

        # Parse ephemeral credentials if provided
        credentials = None
        if request.config.HasField("credentials"):
            from ..sdk.session_manager import SessionCredentials
            cred = request.config.credentials
            credentials = SessionCredentials(
                credential_type=cred.credential_type or "api_key",
                api_key=cred.api_key if cred.HasField("api_key") else None,
                oauth_token=cred.oauth_token if cred.HasField("oauth_token") else None,
                refresh_token=cred.refresh_token if cred.HasField("refresh_token") else None,
                expires_at=cred.expires_at if cred.HasField("expires_at") else None,
            )
            logger.info(f"CreateSession: ephemeral {credentials.credential_type} credentials provided")

        config = SessionConfig(
            project_path=request.config.project_path,
            model=request.config.model or "sonnet",
            permission_mode=request.config.permission_mode or "default",
            allowed_tools=list(request.config.allowed_tools),
            disallowed_tools=list(request.config.disallowed_tools),
            max_turns=request.config.max_turns if request.config.max_turns > 0 else None,
            continue_conversation=request.config.continue_conversation,
            environment=dict(request.config.environment),
            enable_dangerous_command_blocking=request.config.enable_dangerous_command_blocking,
            headless=request.config.headless,
            account_id=request.config.account_id if request.config.HasField("account_id") else None,
            user_email=request.config.user_email if request.config.HasField("user_email") else None,
            credentials=credentials,
            settings_path=request.config.settings_path if request.config.HasField("settings_path") else None,
        )

        try:
            session = await self.session_manager.create_session(config)

            return pb2.Session(
                session_id=session.id,
                session_name=session.name,
                project_path=session.project_path,
                model=session.model,
                status=pb2.SESSION_STATUS_IDLE,
                created_at=session.created_at.isoformat(),
                context_usage=pb2.ContextUsage(
                    current_tokens=session.context_tokens,
                    max_tokens=session.max_context_tokens,
                    percentage=0.0,
                ),
                summary=session.summary or "",
                account_id=session.account_id or "",
                user_email=session.user_email or "",
                has_ephemeral_credentials=session.has_ephemeral_credentials,
            )
        except Exception as e:
            logger.error(f"CreateSession failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetSession(
        self, request: pb2.GetSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb2.Session:
        """Get session information."""
        session = self.session_manager.get_session(request.session_id)

        if not session:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Session not found: {request.session_id}"
            )

        return pb2.Session(
            session_id=session.id,
            session_name=session.name,
            project_path=session.project_path,
            model=session.model,
            status=self._status_to_proto(session.status),
            created_at=session.created_at.isoformat(),
            context_usage=pb2.ContextUsage(
                current_tokens=session.context_tokens,
                max_tokens=session.max_context_tokens,
                percentage=(session.context_tokens / session.max_context_tokens * 100)
                if session.max_context_tokens > 0 else 0.0,
            ),
        )

    async def ListSessions(
        self, request: pb2.ListSessionsRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ListSessionsResponse:
        """List all sessions."""
        sessions = self.session_manager.list_sessions()

        # Apply pagination
        offset = request.offset if request.offset else 0
        limit = request.limit if request.limit else len(sessions)
        paginated = sessions[offset:offset + limit]

        return pb2.ListSessionsResponse(
            sessions=[
                pb2.Session(
                    session_id=s.id,
                    session_name=s.name,
                    project_path=s.project_path,
                    model=s.model,
                    status=self._status_to_proto(s.status),
                    created_at=s.created_at.isoformat(),
                )
                for s in paginated
            ],
            total_count=len(sessions),
        )

    async def DeleteSession(
        self, request: pb2.DeleteSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb2.Empty:
        """Delete a session."""
        logger.info(f"DeleteSession: {request.session_id}")
        await self.session_manager.delete_session(request.session_id)
        return pb2.Empty()

    async def Chat(
        self,
        request_iterator: AsyncIterator[pb2.ChatRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb2.ChatResponse]:
        """Bidirectional streaming chat."""
        async for request in request_iterator:
            session_id = request.session_id
            logger.info(f"Chat request for session: {session_id}")

            # Handle prompt
            if request.HasField("prompt"):
                try:
                    async for msg in self.session_manager.send_prompt(
                        session_id, request.prompt
                    ):
                        yield self._convert_to_proto(session_id, msg)
                except ValueError as e:
                    yield pb2.ChatResponse(
                        session_id=session_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        error=pb2.ErrorInfo(
                            code="SESSION_NOT_FOUND",
                            message=str(e),
                        ),
                    )
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    yield pb2.ChatResponse(
                        session_id=session_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        error=pb2.ErrorInfo(
                            code="INTERNAL_ERROR",
                            message=str(e),
                        ),
                    )

            # Handle command
            elif request.HasField("command"):
                cmd = request.command
                if cmd == "interrupt":
                    try:
                        await self.session_manager.interrupt(session_id)
                        yield pb2.ChatResponse(
                            session_id=session_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            status_update=pb2.SessionStatusUpdate(
                                status=pb2.SESSION_STATUS_IDLE,
                                message="Interrupted",
                            ),
                        )
                    except Exception as e:
                        yield pb2.ChatResponse(
                            session_id=session_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            error=pb2.ErrorInfo(
                                code="INTERRUPT_FAILED",
                                message=str(e),
                            ),
                        )

    async def SendPrompt(
        self, request: pb2.SendPromptRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb2.ChatResponse]:
        """Unary prompt with streaming response (simpler API)."""
        session_id = request.session_id
        prompt = request.prompt
        logger.info(f"SendPrompt for session {session_id}: {len(prompt)} chars")

        try:
            async for msg in self.session_manager.send_prompt(session_id, prompt):
                yield self._convert_to_proto(session_id, msg)
        except ValueError as e:
            yield pb2.ChatResponse(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=pb2.ErrorInfo(
                    code="SESSION_NOT_FOUND",
                    message=str(e),
                ),
            )
        except Exception as e:
            logger.error(f"SendPrompt error: {e}")
            yield pb2.ChatResponse(
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=pb2.ErrorInfo(
                    code="INTERNAL_ERROR",
                    message=str(e),
                ),
            )

    async def Interrupt(
        self, request: pb2.InterruptRequest, context: grpc.aio.ServicerContext
    ) -> pb2.InterruptResponse:
        """Interrupt current execution."""
        logger.info(f"Interrupt: {request.session_id}")
        try:
            await self.session_manager.interrupt(request.session_id)
            return pb2.InterruptResponse(success=True, message="Interrupted successfully")
        except Exception as e:
            return pb2.InterruptResponse(success=False, message=str(e))

    async def Compact(
        self, request: pb2.CompactRequest, context: grpc.aio.ServicerContext
    ) -> pb2.CompactResponse:
        """Compact session context (not yet implemented)."""
        # TODO: Implement context compaction via SDK
        return pb2.CompactResponse(
            success=False,
            summary="Context compaction not yet implemented",
            new_context_usage=pb2.ContextUsage(),
        )

    async def HealthCheck(
        self, request: pb2.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> pb2.HealthCheckResponse:
        """Health check endpoint."""
        return pb2.HealthCheckResponse(
            healthy=True,
            version=SERVER_VERSION,
            active_sessions=self.session_manager.get_active_count(),
            uptime=self.session_manager.get_uptime(),
        )

    async def SubmitAnswer(
        self, request: pb2.SubmitAnswerRequest, context: grpc.aio.ServicerContext
    ) -> pb2.SubmitAnswerResponse:
        """Submit answers to a pending AskUserQuestion."""
        logger.info(f"SubmitAnswer: session={request.session_id}, question={request.question_id}")
        try:
            result = await self.session_manager.submit_answer(
                request.session_id,
                request.question_id,
                dict(request.answers),
            )
            return pb2.SubmitAnswerResponse(
                success=result.get("success", False),
                error=result.get("error"),
            )
        except Exception as e:
            logger.error(f"SubmitAnswer error: {e}")
            return pb2.SubmitAnswerResponse(success=False, error=str(e))

    async def GetPendingQuestion(
        self, request: pb2.GetPendingQuestionRequest, context: grpc.aio.ServicerContext
    ) -> pb2.GetPendingQuestionResponse:
        """Get the pending question for a session, if any."""
        logger.info(f"GetPendingQuestion: session={request.session_id}")

        pending = self.session_manager.get_pending_question(request.session_id)
        if not pending:
            return pb2.GetPendingQuestionResponse(has_question=False)

        # Build Question proto from pending data
        questions = pending.questions
        if questions:
            first_q = questions[0]
            question_proto = pb2.Question(
                question_id=pending.question_id,
                question_text=first_q.get("question", ""),
                options=[
                    pb2.QuestionOption(
                        id=str(i),
                        label=opt.get("label", ""),
                        description=opt.get("description", ""),
                    )
                    for i, opt in enumerate(first_q.get("options", []))
                ],
                allow_custom=first_q.get("allow_custom", True),
            )
            return pb2.GetPendingQuestionResponse(
                has_question=True,
                question=question_proto,
                tool_use_id=pending.tool_use_id,
            )

        return pb2.GetPendingQuestionResponse(has_question=False)

    def _convert_to_proto(
        self, session_id: str, msg: StreamMessage
    ) -> pb2.ChatResponse:
        """Convert StreamMessage to protobuf ChatResponse."""
        response = pb2.ChatResponse(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if msg.type == "text":
            response.text.CopyFrom(pb2.TextChunk(
                content=msg.content or "",
                is_complete=True,
            ))
        elif msg.type == "thinking":
            response.thinking.CopyFrom(pb2.ThinkingChunk(
                content=msg.content or "",
            ))
        elif msg.type == "tool_use":
            response.tool_use.CopyFrom(pb2.ToolUse(
                tool_use_id=msg.tool_id or "",
                name=msg.tool_name or "",
                input_json=json.dumps(msg.tool_input) if msg.tool_input else "{}",
            ))
        elif msg.type == "tool_result":
            response.tool_result.CopyFrom(pb2.ToolResult(
                tool_use_id=msg.tool_id or "",
                content=msg.content or "",
                is_error=msg.is_error,
            ))
        elif msg.type == "cost":
            response.cost.CopyFrom(pb2.CostInfo(
                total_cost_usd=msg.cost_usd or 0.0,
                input_tokens=msg.input_tokens or 0,
                output_tokens=msg.output_tokens or 0,
            ))
        elif msg.type == "status":
            status_enum = self._status_str_to_proto(msg.status or "idle")
            response.status_update.CopyFrom(pb2.SessionStatusUpdate(
                status=status_enum,
                message=msg.content,
            ))
        elif msg.type == "error":
            response.error.CopyFrom(pb2.ErrorInfo(
                code="SDK_ERROR",
                message=msg.content or "Unknown error",
            ))
        elif msg.type == "question":
            # Convert question to proto
            options = []
            if msg.question_options:
                for i, q in enumerate(msg.question_options):
                    for j, opt in enumerate(q.get("options", [])):
                        options.append(pb2.QuestionOption(
                            id=f"{i}_{j}",
                            label=opt.get("label", ""),
                            description=opt.get("description", ""),
                        ))
            response.question.CopyFrom(pb2.Question(
                question_id=msg.question_id or "",
                question_text=msg.question_text or "",
                options=options,
                allow_custom=msg.allow_custom,
            ))
        elif msg.type == "blocked_command":
            # Convert blocked command to error with details
            response.error.CopyFrom(pb2.ErrorInfo(
                code="BLOCKED_COMMAND",
                message=msg.content or "Command blocked",
                details=f"Category: {msg.blocked_category}, Pattern: {msg.blocked_pattern}",
            ))
        elif msg.type == "subagent_complete":
            # Subagent (Task tool) completed
            response.subagent_complete.CopyFrom(pb2.SubagentComplete(
                agent_id=msg.agent_id or "",
                result=msg.content or "",
                duration_ms=msg.agent_duration_ms or 0.0,
                is_error=msg.is_error,
            ))

        return response

    def _status_to_proto(self, status: str) -> int:
        """Convert status string to protobuf enum."""
        return self._status_str_to_proto(status)

    def _status_str_to_proto(self, status: str) -> int:
        """Convert status string to protobuf enum value."""
        mapping = {
            "pending": pb2.SESSION_STATUS_PENDING,
            "running": pb2.SESSION_STATUS_RUNNING,
            "idle": pb2.SESSION_STATUS_IDLE,
            "completed": pb2.SESSION_STATUS_COMPLETED,
            "error": pb2.SESSION_STATUS_ERROR,
            "terminated": pb2.SESSION_STATUS_TERMINATED,
            "waiting_for_input": pb2.SESSION_STATUS_WAITING_FOR_INPUT,
            "compacting": pb2.SESSION_STATUS_RUNNING,  # Map compacting to running
        }
        return mapping.get(status.lower(), pb2.SESSION_STATUS_UNSPECIFIED)

    # --- Session Summary ---

    async def GenerateSummary(
        self, request: pb2.GenerateSummaryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.GenerateSummaryResponse:
        """Generate an AI summary of session activity."""
        logger.info(f"GenerateSummary: session={request.session_id}")

        max_chars = request.max_chars if request.max_chars > 0 else 150
        result = await self.session_manager.generate_summary(request.session_id, max_chars)

        return pb2.GenerateSummaryResponse(
            success=result.get("success", False),
            summary=result.get("summary"),
            error=result.get("error"),
        )

    # --- OAuth & Account Management ---

    async def GetAuthStatus(
        self, request: pb2.GetAuthStatusRequest, context: grpc.aio.ServicerContext
    ) -> pb2.GetAuthStatusResponse:
        """Get CLI authentication status.

        Checks if the Claude Code CLI is authenticated.
        """
        logger.info("GetAuthStatus: checking CLI authentication")

        status = self.session_manager.get_auth_status()
        return pb2.GetAuthStatusResponse(
            authenticated=status.get("authenticated", False),
            account_type=status.get("account_type", ""),
            email=status.get("email", ""),
            error=status.get("error", ""),
        )

    async def StartOAuth(
        self, request: pb2.StartOAuthRequest, context: grpc.aio.ServicerContext
    ) -> pb2.StartOAuthResponse:
        """Start OAuth authentication flow.

        For Claude Code, OAuth is managed by the CLI. This method returns
        instructions for authenticating via the CLI rather than an authorization URL.
        """
        logger.info(f"StartOAuth: redirect_uri={request.redirect_uri}")

        result = self.session_manager.start_oauth(request.redirect_uri)

        # Handle both success (already authenticated) and requires_cli_login cases
        if result.get("success"):
            return pb2.StartOAuthResponse(
                success=True,
                message=result.get("message", "Already authenticated"),
                email=result.get("email", ""),
            )
        else:
            return pb2.StartOAuthResponse(
                success=False,
                requires_cli_login=result.get("requires_cli_login", False),
                instructions=result.get("instructions", result.get("error", "")),
            )

    async def CompleteOAuth(
        self, request: pb2.CompleteOAuthRequest, context: grpc.aio.ServicerContext
    ) -> pb2.CompleteOAuthResponse:
        """Complete OAuth flow - not applicable for CLI-based auth.

        The Claude Code CLI handles OAuth internally. This endpoint exists
        for API compatibility but returns guidance to use CLI authentication.
        """
        logger.info(f"CompleteOAuth: state={request.state[:8] if request.state else 'none'}...")

        result = await self.session_manager.complete_oauth(request.code, request.state)
        return pb2.CompleteOAuthResponse(
            success=result.get("success", False),
            error=result.get("error"),
            account_id=result.get("account_id"),
        )

    async def RefreshToken(
        self, request: pb2.RefreshTokenRequest, context: grpc.aio.ServicerContext
    ) -> pb2.RefreshTokenResponse:
        """Refresh/validate OAuth token for an account.

        For CLI-based OAuth, token refresh is handled by the CLI. This method
        checks if the CLI authentication is still valid.
        """
        logger.info(f"RefreshToken: account_id={request.account_id}")

        result = await self.session_manager.refresh_token(request.account_id)

        return pb2.RefreshTokenResponse(
            success=result.get("success", False),
            error=result.get("error"),
            message=result.get("message", ""),
            requires_reauth=result.get("requires_reauth", False),
            instructions=result.get("instructions", ""),
        )

    async def SetAccountCredentials(
        self, request: pb2.SetAccountCredentialsRequest, context: grpc.aio.ServicerContext
    ) -> pb2.SetAccountCredentialsResponse:
        """Set credentials for an account."""
        logger.info(f"SetAccountCredentials: account_id={request.account_id}")

        if request.HasField("api_key"):
            result = self.session_manager.set_account_credentials(
                account_id=request.account_id,
                credential_type="api_key",
                api_key=request.api_key.api_key,
            )
        elif request.HasField("oauth"):
            result = self.session_manager.set_account_credentials(
                account_id=request.account_id,
                credential_type="oauth",
                access_token=request.oauth.access_token,
                refresh_token=request.oauth.refresh_token,
                expires_at=request.oauth.expires_at,
            )
        else:
            result = {"success": False, "error": "No credentials provided"}

        return pb2.SetAccountCredentialsResponse(
            success=result.get("success", False),
            error=result.get("error"),
        )

    async def GetAccountStatus(
        self, request: pb2.GetAccountStatusRequest, context: grpc.aio.ServicerContext
    ) -> pb2.GetAccountStatusResponse:
        """Get status of an account's credentials."""
        logger.info(f"GetAccountStatus: account_id={request.account_id}")

        status = self.session_manager.get_account_status(request.account_id)

        return pb2.GetAccountStatusResponse(
            exists=status.get("exists", False),
            credential_type=status.get("credential_type", ""),
            is_valid=status.get("is_valid", False),
            token_expires_at=status.get("token_expires_at"),
        )
