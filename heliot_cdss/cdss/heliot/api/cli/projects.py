from __future__ import annotations

import argparse
import sys
from typing import Iterable

from sqlalchemy.orm import Session

from ..db.session import SessionLocal
from ..models.project_db import Project
from ..services.project_service import (
    ProjectAlreadyExistsError,
    ProjectCreate,
    ProjectNotFoundError,
    ProjectService,
)


def _open_session() -> Session:
    return SessionLocal()


def _print_project(p: Project) -> None:
    print(f"id={p.id} name={p.name!r} active={p.is_active} created_at={p.created_at} updated_at={p.updated_at}")


def cmd_create(args: argparse.Namespace) -> int:
    with _open_session() as db:
        svc = ProjectService(db)
        try:
            p = svc.create(ProjectCreate(name=args.name, is_active=not args.inactive))
        except ProjectAlreadyExistsError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        _print_project(p)
        return 0


def cmd_get(args: argparse.Namespace) -> int:
    with _open_session() as db:
        svc = ProjectService(db)
        try:
            p = svc.require_by_id(args.id)
        except ProjectNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3

        _print_project(p)
        return 0


def cmd_get_by_name(args: argparse.Namespace) -> int:
    with _open_session() as db:
        svc = ProjectService(db)
        try:
            p = svc.require_by_name(args.name)
        except ProjectNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3

        _print_project(p)
        return 0


def cmd_list(args: argparse.Namespace) -> int:
    with _open_session() as db:
        q = db.query(Project).order_by(Project.id.asc())
        if args.active_only:
            q = q.filter(Project.is_active.is_(True))
        if args.name_contains:
            # Simple contains filter; if you want case-insensitive use ilike().
            q = q.filter(Project.name.contains(args.name_contains))

        projects = q.all()
        if not projects:
            print("(no projects)")
            return 0

        for p in projects:
            _print_project(p)
        return 0


def cmd_set_active(args: argparse.Namespace) -> int:
    with _open_session() as db:
        svc = ProjectService(db)

        active_str = args.active.strip().lower()
        if active_str in {"true", "1", "yes", "y"}:
            is_active = True
        elif active_str in {"false", "0", "no", "n"}:
            is_active = False
        else:
            print("ERROR: --active must be true/false", file=sys.stderr)
            return 2

        try:
            p = svc.set_active(project_id=args.id, is_active=is_active)
        except ProjectNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3

        _print_project(p)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heliot-projects", description="Manage Heliot Projects.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new project.")
    p_create.add_argument("--name", required=True, help="Unique project name.")
    p_create.add_argument(
        "--inactive",
        action="store_true",
        help="Create project as inactive (default: active).",
    )
    p_create.set_defaults(func=cmd_create)

    p_get = sub.add_parser("get", help="Get project by id.")
    p_get.add_argument("--id", type=int, required=True)
    p_get.set_defaults(func=cmd_get)

    p_getn = sub.add_parser("get-by-name", help="Get project by name.")
    p_getn.add_argument("--name", required=True)
    p_getn.set_defaults(func=cmd_get_by_name)

    p_list = sub.add_parser("list", help="List projects.")
    p_list.add_argument("--active-only", action="store_true", help="Show only active projects.")
    p_list.add_argument("--name-contains", default=None, help="Filter by substring match on name.")
    p_list.set_defaults(func=cmd_list)

    p_set = sub.add_parser("set-active", help="Activate/deactivate a project.")
    p_set.add_argument("--id", type=int, required=True)
    p_set.add_argument("--active", required=True, help="true/false")
    p_set.set_defaults(func=cmd_set_active)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())