import hashlib
import hmac
import secrets

from fastapi import Request, Response

from app.core.config import settings


def _sign(workspace_id: str) -> str:
    return hmac.new(settings.workspace_cookie_secret.encode(), workspace_id.encode(), hashlib.sha256).hexdigest()


def _encode(workspace_id: str) -> str:
    return f"{workspace_id}.{_sign(workspace_id)}"


def _decode(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    workspace_id, signature = value.rsplit(".", 1)
    if hmac.compare_digest(signature, _sign(workspace_id)):
        return workspace_id
    return None


def workspace_from_request(request: Request) -> str:
    existing = getattr(request.state, "workspace_id", None)
    if existing:
        return existing
    workspace_id = _decode(request.cookies.get(settings.workspace_cookie_name))
    if not workspace_id:
        workspace_id = secrets.token_urlsafe(24)
        request.state.workspace_cookie = _encode(workspace_id)
    request.state.workspace_id = workspace_id
    return workspace_id


def attach_workspace_cookie(request: Request, response: Response) -> None:
    cookie = getattr(request.state, "workspace_cookie", None)
    if cookie:
        response.set_cookie(
            settings.workspace_cookie_name,
            cookie,
            httponly=True,
            samesite="lax",
            secure=settings.environment == "production",
            max_age=60 * 60 * 24 * 365,
        )
