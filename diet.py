import datetime
import json

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from .prompts import DIET_SYSTEM_PROMPT
from .schema import DietRequest, DietResponse
from .storage import get_current_session_id, remember_meal


def default_diet_timestamp(diet_request: DietRequest) -> DietRequest:
    now = datetime.datetime.now()
    return diet_request.model_copy(update={
        "date": diet_request.date or now.date().isoformat(),
        "time": diet_request.time or now.strftime("%H:%M"),
    })


def run_diet_agent(
    diet_request: DietRequest,
    model,
    search_tool,
    session_id: str | None = None,
) -> DietResponse:
    diet_request = default_diet_timestamp(diet_request)
    diet_agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=DIET_SYSTEM_PROMPT,
        response_format=DietResponse,
    )
    result = diet_agent.invoke({
        "messages": [
            HumanMessage(content=json.dumps(diet_request.model_dump()))
        ]
    })
    diet_response = result.get("structured_response")

    if diet_response is None:
        raise ValueError("diet agent did not return structured output")

    response = DietResponse.model_validate(diet_response)
    remember_meal(session_id or get_current_session_id(), diet_request, response)

    return response
