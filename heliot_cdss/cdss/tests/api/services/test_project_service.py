from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError

fake_models_module = ModuleType("cdss.heliot.api.models.project_db")


class FakeProject:
    name = "name"
    is_active = "is_active"

    def __init__(self, name: str, is_active: bool):
        self.name = name
        self.is_active = is_active


fake_models_module.Project = FakeProject

sys.modules["cdss.heliot.api.models.project_db"] = fake_models_module

from cdss.heliot.api.services.project_service import (  # noqa: E402
    ProjectService,
    ProjectCreate,
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
)


def _make_db_mock() -> Mock:
    """Create a Session-like mock with common methods used by ProjectService."""
    db = Mock(name="db_session")
    db.get = Mock(name="get")
    db.add = Mock(name="add")
    db.commit = Mock(name="commit")
    db.rollback = Mock(name="rollback")
    db.refresh = Mock(name="refresh")
    db.query = Mock(name="query")
    return db


class TestProjectServiceGetById:
    def test_returns_project_when_found(self):
        """
        GOAL: get_by_id should return the project returned by Session.get().
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        project = SimpleNamespace(id=1, name="acme", is_active=True)
        db.get.return_value = project

        # Act
        result = service.get_by_id(1)

        # Assert
        assert result is project
        assert db.get.call_count == 1
        _, called_id = db.get.call_args.args
        assert called_id == 1

    def test_returns_none_when_not_found(self):
        """
        GOAL: get_by_id should return None when the project does not exist.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        db.get.return_value = None

        # Act
        result = service.get_by_id(999)

        # Assert
        assert result is None
        assert db.get.call_count == 1


class TestProjectServiceGetByName:
    def test_returns_none_on_blank_name(self):
        """
        GOAL: get_by_name should return None for blank/whitespace input without querying the DB.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        # Act
        result = service.get_by_name("   ")

        # Assert
        assert result is None
        db.query.assert_not_called()

    def test_strips_name_and_queries_db(self):
        """
        GOAL: get_by_name should strip the input and query the DB for an exact match.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        query = Mock(name="query")
        filtered = Mock(name="filtered")
        db.query.return_value = query
        query.filter.return_value = filtered

        project = SimpleNamespace(id=1, name="acme", is_active=True)
        filtered.one_or_none.return_value = project

        # Act
        result = service.get_by_name("  acme  ")

        # Assert
        assert result is project
        db.query.assert_called_once()
        query.filter.assert_called_once()
        filtered.one_or_none.assert_called_once_with()


class TestProjectServiceRequireById:
    def test_returns_project_when_found(self):
        """
        GOAL: require_by_id should return the project when present.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        project = SimpleNamespace(id=1, name="acme", is_active=True)
        db.get.return_value = project

        # Act
        result = service.require_by_id(1)

        # Assert
        assert result is project

    def test_raises_when_not_found(self):
        """
        GOAL: require_by_id should raise ProjectNotFoundError when missing.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        db.get.return_value = None

        # Act / Assert
        with pytest.raises(ProjectNotFoundError, match=r"Project id=123 not found"):
            service.require_by_id(123)


class TestProjectServiceRequireByName:
    def test_returns_project_when_found(self):
        """
        GOAL: require_by_name should return the project when present.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        query = Mock()
        filtered = Mock()
        db.query.return_value = query
        query.filter.return_value = filtered

        project = SimpleNamespace(id=1, name="acme", is_active=True)
        filtered.one_or_none.return_value = project

        # Act
        result = service.require_by_name("acme")

        # Assert
        assert result is project

    def test_raises_when_not_found_including_blank(self):
        """
        GOAL: require_by_name should raise ProjectNotFoundError when missing (including blank input).
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        # Act / Assert
        with pytest.raises(ProjectNotFoundError, match=r"Project name='   ' not found"):
            service.require_by_name("   ")


class TestProjectServiceCreate:
    def test_raises_value_error_on_empty_name(self):
        """
        GOAL: create should reject empty/whitespace project names.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        data = ProjectCreate(name="   ")

        # Act / Assert
        with pytest.raises(ValueError, match="Project name cannot be empty"):
            service.create(data)

        db.add.assert_not_called()
        db.commit.assert_not_called()
        db.refresh.assert_not_called()

    def test_raises_value_error_on_name_too_long(self):
        """
        GOAL: create should reject names longer than 200 characters.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        data = ProjectCreate(name="a" * 201)

        # Act / Assert
        with pytest.raises(ValueError, match="Project name too long"):
            service.create(data)

        db.add.assert_not_called()
        db.commit.assert_not_called()
        db.refresh.assert_not_called()

    def test_commits_and_refreshes_on_success(self):
        """
        GOAL: create should add, commit, refresh and return the created project on success.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        data = ProjectCreate(name="  acme  ", is_active=True)

        # Act
        project = service.create(data)

        # Assert
        assert project.name == "acme"
        assert project.is_active is True

        db.add.assert_called_once()
        added_obj = db.add.call_args.args[0]
        assert added_obj is project

        db.commit.assert_called_once_with()
        db.refresh.assert_called_once_with(project)
        db.rollback.assert_not_called()

    def test_rolls_back_and_raises_on_unique_violation(self):
        """
        GOAL: create should rollback and raise ProjectAlreadyExistsError on IntegrityError.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        db.commit.side_effect = IntegrityError("stmt", "params", Exception("orig"))

        data = ProjectCreate(name="acme")

        # Act / Assert
        with pytest.raises(ProjectAlreadyExistsError, match=r"Project name='acme' already exists"):
            service.create(data)

        db.add.assert_called_once()
        db.commit.assert_called_once_with()
        db.rollback.assert_called_once_with()
        db.refresh.assert_not_called()


class TestProjectServiceSetActive:
    def test_sets_flag_commits_refreshes(self):
        """
        GOAL: set_active should update is_active, persist changes, and return the updated project.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)

        project = SimpleNamespace(id=1, name="acme", is_active=True)
        db.get.return_value = project

        # Act
        result = service.set_active(project_id=1, is_active=False)

        # Assert
        assert result is project
        assert project.is_active is False
        db.add.assert_called_once_with(project)
        db.commit.assert_called_once_with()
        db.refresh.assert_called_once_with(project)

    def test_raises_when_project_missing(self):
        """
        GOAL: set_active should raise ProjectNotFoundError when the project does not exist.
        """
        # Arrange
        db = _make_db_mock()
        service = ProjectService(db)
        db.get.return_value = None

        # Act / Assert
        with pytest.raises(ProjectNotFoundError, match=r"Project id=1 not found"):
            service.set_active(project_id=1, is_active=False)

        db.add.assert_not_called()
        db.commit.assert_not_called()
        db.refresh.assert_not_called()