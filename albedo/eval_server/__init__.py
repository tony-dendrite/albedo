"""albedo.eval_server — FastAPI evaluation server for Albedo subnet."""
from __future__ import annotations

from albedo.eval_server.endpoints import app, main

__all__ = ["app", "main"]
