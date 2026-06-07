import datetime
import uuid
from contextvars import ContextVar
from typing import Any

from langchain.messages import AIMessage, HumanMessage

from .schema import DietRequest, DietResponse, ProgramRequest, ProgramResponse


MAX_CONVERSATION_MESSAGES = 30
CONVERSATIONS: dict[str, list] = {}
MEALS: dict[str, list[dict[str, Any]]] = {}
PROGRAMS: dict[str, list[dict[str, Any]]] = {}
CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar("CURRENT_SESSION_ID", default=None)


def get_optional_session_id(data: dict) -> str | None:
    session_id = data.get("session_id")

    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()

    return None


def get_session_id(data: dict) -> str:
    return get_optional_session_id(data) or str(uuid.uuid4())


def get_current_session_id() -> str | None:
    return CURRENT_SESSION_ID.get()


def get_conversation_messages(session_id: str) -> list:
    return CONVERSATIONS.setdefault(session_id, [])


def remember_conversation_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    messages = get_conversation_messages(session_id)
    messages.extend([
        HumanMessage(content=user_message),
        AIMessage(content=assistant_message),
    ])
    CONVERSATIONS[session_id] = messages[-MAX_CONVERSATION_MESSAGES:]


def remember_meal(session_id: str | None, diet_request: DietRequest, diet_response: DietResponse) -> None:
    if not session_id:
        return

    MEALS.setdefault(session_id, []).append({
        "created_at": datetime.datetime.now().isoformat(),
        "request": diet_request.model_dump(),
        "response": diet_response.model_dump(),
    })


def remember_program(session_id: str | None, program_request: ProgramRequest, program_response: ProgramResponse) -> None:
    if not session_id:
        return

    PROGRAMS.setdefault(session_id, []).append({
        "created_at": datetime.datetime.now().isoformat(),
        "request": program_request.model_dump(),
        "response": program_response.model_dump(),
    })
