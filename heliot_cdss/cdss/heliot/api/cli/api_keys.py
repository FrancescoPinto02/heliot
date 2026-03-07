# cdss/heliot/api/cli/api_keys.py
"""
Minimal CLI for managing API keys.

Examples:
  # Create a key (prints token ONCE)
  poetry run python -m cdss.heliot.api.cli.api_keys create --project-name "acme" --env prod --name "default"

  # Create a key by project id
  poetry run python -m cdss.heliot.api.cli.api_keys create --project-id 1 --env test --name "default1"

  # List keys
  poetry run python -m cdss.heliot.api.cli.api_keys list --project-name "acme"
  poetry run python -m cdss.heliot.api.cli.api_keys list --project-id 1 --active-only

  # Revoke key by prefix
  python -m cdss.heliot.api.cli.api_keys revoke --prefix 8fj29dk3qz

Env requirements:
  - DATABASE_URL must be set (your db/session.py enforces it)
  - HELIOT_API_KEY_PEPPER_V1 must be set (recommended base64url). Example:
        export HELIOT_API_KEY_PEPPER_V1="pQ7...base64url..."
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..db.session import SessionLocal
from ..models.api_key_db import ApiKey
from ..security.pepper import EnvPepperProvider
from ..services.api_key_service import ApiKeyCreate, ApiKeyService, ApiKeyError
from ..services.project_service import ProjectService, ProjectNotFoundError


def _open_session() -> Session:
    return SessionLocal()


def _mask_token(env: str, prefix: str) -> str:
    # Never print the secret. Only show a recognizable masked form.
    return f"hl_sk_{env}_{prefix}_***"


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    # Keep it ISO-like and explicit in UTC when possible
    if dt.tzinfo is None:
        return dt.isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def _resolve_project_id(db: Session, project_id: int | None, project_name: str | None) -> int:
    svc = ProjectService(db)

    if project_id is not None:
        p = svc.get_by_id(project_id)
        if p is None:
            raise ProjectNotFoundError(f"Project id={project_id} not found")
        return p.id

    if project_name:
        p = svc.get_by_name(project_name)
        if p is None:
            raise ProjectNotFoundError(f"Project name='{project_name}' not found")
        return p.id

    raise ValueError("Either --project-id or --project-name is required")


def _parse_scopes(scopes_raw: str | None) -> list[str]:
    if not scopes_raw:
        return []
    scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()]
    # Deduplicate while preserving order
    seen = set()
    out: list[str] = []
    for s in scopes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_expires_at(expires_at_raw: str | None) -> datetime | None:
    """
    Accepts ISO-8601 timestamps. If no timezone is provided, assume UTC (safe default).
    Examples:
      2026-06-01T12:00:00Z
      2026-06-01T12:00:00+00:00
      2026-06-01T12:00:00
    """
    if not expires_at_raw:
        return None

    s = expires_at_raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def cmd_create(args: argparse.Namespace) -> int:
    with _open_session() as db:
        try:
            project_id = _resolve_project_id(db, args.project_id, args.project_name)
        except (ProjectNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        pepper_provider = EnvPepperProvider()
        svc = ApiKeyService(db, pepper_provider)

        scopes = _parse_scopes(args.scopes)
        expires_at = _parse_expires_at(args.expires_at)

        try:
            token_plain, api_key = svc.create(
                ApiKeyCreate(
                    project_id=project_id,
                    env=args.env,
                    name=args.name,
                    description=args.description,
                    scopes=scopes,
                    expires_at=expires_at,
                    pepper_version=args.pepper_version,
                    prefix_len=args.prefix_len,
                    secret_bytes=args.secret_bytes,
                )
            )
        except ApiKeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            # Unexpected errors (keep message minimal)
            print(f"ERROR: Unexpected failure: {e}", file=sys.stderr)
            return 1

        # Print metadata + token ONCE
        print("API KEY CREATED")
        print(f"  api_key_id:   {api_key.id}")
        print(f"  project_id:   {api_key.project_id}")
        print(f"  key_prefix:   {api_key.key_prefix}")
        print(f"  env:          {args.env}")
        print(f"  scopes:       {','.join(api_key.scopes or []) or '-'}")
        print(f"  expires_at:   {_fmt_dt(api_key.expires_at)}")
        print("")
        print("TOKEN (SHOWING ONCE):")
        print(token_plain)
        return 0


def cmd_list(args: argparse.Namespace) -> int:
    with _open_session() as db:
        try:
            project_id = _resolve_project_id(db, args.project_id, args.project_name)
        except (ProjectNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        pepper_provider = EnvPepperProvider()
        svc = ApiKeyService(db, pepper_provider)

        keys = svc.list_by_project(project_id, include_inactive=not args.active_only)
        if not keys:
            print("(no api keys)")
            return 0

        for k in keys:
            status = "active"
            if not k.is_active:
                status = "inactive"
            if k.revoked_at is not None:
                status = "revoked"
            # env is not stored in DB; we cannot reconstruct it reliably from prefix alone.
            # We display a generic masked string without env.
            masked = _mask_token(k.env, k.key_prefix)

            print(
                " ".join(
                    [
                        f"id={k.id}",
                        f"project_id={k.project_id}",
                        f"key={masked}",
                        f"status={status}",
                        f"scopes={','.join(k.scopes or []) or '-'}",
                        f"expires_at={_fmt_dt(k.expires_at)}",
                        f"last_used_at={_fmt_dt(k.last_used_at)}",
                        f"created_at={_fmt_dt(k.created_at)}",
                        (f"name={k.name!r}" if k.name else "name=-"),
                    ]
                )
            )

        return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    with _open_session() as db:
        pepper_provider = EnvPepperProvider()
        svc = ApiKeyService(db, pepper_provider)

        try:
            k = svc.revoke_by_prefix(args.prefix)
        except ApiKeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        print("API KEY REVOKED")
        print(f"  id:         {k.id}")
        print(f"  project_id: {k.project_id}")
        print(f"  key:        {_mask_token(k.env, k.key_prefix)}")
        print(f"  revoked_at: {_fmt_dt(k.revoked_at)}")
        return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """
    Optional helper command for local testing: verifies a token against the DB+pepper.
    Does NOT print secrets; prints only the result.
    """
    with _open_session() as db:
        pepper_provider = EnvPepperProvider()
        svc = ApiKeyService(db, pepper_provider)

        ctx = svc.verify(args.token, update_last_used=False)
        if ctx is None:
            print("INVALID")
            return 4

        print("VALID")
        print(f"  project_id: {ctx.project_id}")
        print(f"  api_key_id: {ctx.api_key_id}")
        print(f"  key_prefix: {ctx.key_prefix}")
        print(f"  env:        {ctx.env}")
        print(f"  scopes:     {','.join(ctx.scopes) or '-'}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heliot-api-keys", description="Manage Heliot API keys.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new API key (prints token once).")
    g = p_create.add_mutually_exclusive_group(required=True)
    g.add_argument("--project-id", type=int, help="Project id")
    g.add_argument("--project-name", help="Project name")

    p_create.add_argument("--env", required=True, choices=["test", "prod"], help="Token environment")
    p_create.add_argument("--name", default=None, help="Optional key name")
    p_create.add_argument("--description", default=None, help="Optional key description")
    p_create.add_argument("--scopes", default=None, help="Comma-separated scopes (e.g. read,write)")
    p_create.add_argument("--expires-at", default=None, help="ISO-8601 datetime (default: no expiry)")
    p_create.add_argument("--pepper-version", type=int, default=1, help="Pepper version (default: 1)")
    p_create.add_argument("--prefix-len", type=int, default=12, help="Public prefix length (default: 12)")
    p_create.add_argument("--secret-bytes", type=int, default=32, help="Secret size in bytes (default: 32)")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List API keys for a project (no secrets).")
    g2 = p_list.add_mutually_exclusive_group(required=True)
    g2.add_argument("--project-id", type=int, help="Project id")
    g2.add_argument("--project-name", help="Project name")
    p_list.add_argument("--active-only", action="store_true", help="Show only active (not revoked) keys")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke", help="Revoke an API key by prefix.")
    p_revoke.add_argument("--prefix", required=True, help="Public key prefix")
    p_revoke.set_defaults(func=cmd_revoke)

    # Optional helper for local testing
    p_verify = sub.add_parser("verify", help="Verify a token against DB (testing helper).")
    p_verify.add_argument("--token", required=True, help="Full token (do not paste into shared logs)")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())