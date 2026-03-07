from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..security.pepper import EnvPepperProvider
from ..services.api_key_service import ApiKeyService, AuthContext


def require_api_key(authorization: str | None = Header(default=None, alias="Authorization"), db: Session = Depends(get_db)) -> AuthContext:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Expect: "Bearer <token>"
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
        )

    token = parts[1].strip()

    svc = ApiKeyService(db, EnvPepperProvider())
    ctx = svc.verify(token, update_last_used=False)
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return ctx