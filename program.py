import json

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from .prompts import PROGRAM_SYSTEM_PROMPT
from .schema import ProgramRequest, ProgramResponse
from .storage import get_current_session_id, remember_program


def run_program_agent(
    program_request: ProgramRequest,
    model,
    search_tool,
    session_id: str | None = None,
) -> ProgramResponse:
    program_agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=PROGRAM_SYSTEM_PROMPT,
        response_format=ProgramResponse,
    )
    result = program_agent.invoke({
        "messages": [
            HumanMessage(content=json.dumps(program_request.model_dump()))
        ]
    })
    program_response = result.get("structured_response")

    if program_response is None:
        raise ValueError("program agent did not return structured output")

    response = ProgramResponse.model_validate(program_response)
    remember_program(session_id or get_current_session_id(), program_request, response)

    return response
