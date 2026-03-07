from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.project_db import Project


class ProjectAlreadyExistsError(Exception):
    """Raised when attempting to create a project with a name that already exists."""


class ProjectNotFoundError(Exception):
    """Raised when a requested project does not exist."""


@dataclass(frozen=True, slots=True)
class ProjectCreate:
    """Input DTO for creating a project."""
    name: str
    is_active: bool = True


class ProjectService:
    """
    Service layer for Project operations.

    Usage:
        service = ProjectService(db_session)
        project = service.create(ProjectCreate(name="acme"))
    """

    def __init__(self, db: Session):
        self._db = db

    def get_by_id(self, project_id: int) -> Optional[Project]:
        """Return a Project by id, or None if not found."""
        return self._db.get(Project, project_id)

    def get_by_name(self, name: str) -> Optional[Project]:
        """Return a Project by unique name (case-sensitive as stored), or None if not found."""
        name = name.strip()
        if not name:
            return None

        return (
            self._db.query(Project)
            .filter(Project.name == name)
            .one_or_none()
        )

    def require_by_id(self, project_id: int) -> Project:
        """Return a Project by id, raising if not found."""
        project = self.get_by_id(project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project id={project_id} not found")
        return project

    def require_by_name(self, name: str) -> Project:
        """Return a Project by name, raising if not found."""
        project = self.get_by_name(name)
        if project is None:
            raise ProjectNotFoundError(f"Project name='{name}' not found")
        return project

    def create(self, data: ProjectCreate) -> Project:
        """
        Create a new project.

        Raises:
            ProjectAlreadyExistsError: if a project with the same name already exists.
            ValueError: if the input is invalid.
        """
        name = data.name.strip()
        if not name:
            raise ValueError("Project name cannot be empty")
        if len(name) > 200:
            raise ValueError("Project name too long (max 200 chars)")

        project = Project(name=name, is_active=bool(data.is_active))
        self._db.add(project)

        try:
            self._db.commit()
        except IntegrityError:
            # Unique constraint violation (projects.name)
            self._db.rollback()
            raise ProjectAlreadyExistsError(f"Project name='{name}' already exists")

        self._db.refresh(project)
        return project

    def set_active(self, project_id: int, is_active: bool) -> Project:
        """
        Activate/deactivate a project.

        Note:
            Deactivating a project should effectively disable all its keys
            during token verification (checked in ApiKeyService).

        Raises:
            ProjectNotFoundError: if the project does not exist.
        """
        project = self.require_by_id(project_id)
        project.is_active = bool(is_active)
        self._db.add(project)
        self._db.commit()
        self._db.refresh(project)
        return project