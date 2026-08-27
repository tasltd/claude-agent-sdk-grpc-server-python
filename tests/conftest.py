"""Test configuration for the Python gRPC server.

The package under test is a plain library import — no network, no gRPC channel,
no database. Tests stub ``ClaudeSDKClient`` rather than starting a CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
