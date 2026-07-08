"""`albedo` entrypoint - one binary, three roles: backend, gpu-eval, gpu-sanity."""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger


def main() -> None:
    # Parses the role and hands off to the matching process entry.
    parser = argparse.ArgumentParser(prog="albedo", description="Albedo SN97 validator stack")
    sub = parser.add_subparsers(dest="role", required=True)
    sub.add_parser("backend", help="all CPU-side orchestration loops in one process")
    sub.add_parser("gpu-eval", help="8-GPU duel worker API (run on the eval GPU host)")
    sub.add_parser("gpu-sanity", help="sanity pre-eval worker API (run on the sanity GPU host)")
    migrate = sub.add_parser("migrate", help="apply schema.sql to the configured database")
    migrate.add_argument("--schema", default="schema.sql")
    args = parser.parse_args()

    logger.info(f"[cli] starting role={args.role}")
    if args.role == "backend":
        from albedo.backend import run_backend

        asyncio.run(run_backend())
    elif args.role == "gpu-eval":
        from albedo.remote.eval_worker import run_server

        run_server()
    elif args.role == "gpu-sanity":
        from albedo.remote.sanity_worker import run_server

        run_server()
    elif args.role == "migrate":
        from albedo.migrate import apply_schema

        asyncio.run(apply_schema(args.schema))


if __name__ == "__main__":
    main()
