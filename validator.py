"""Validator entrypoint — keeps PM2/ecosystem.config.js pointing here."""
import asyncio
from albedo.validator import main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
