import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "fitness"

from .diet import run_diet_agent
from .program import run_program_agent
from .prompts import CHAT_SYSTEM_PROMPT
from .schema import DietRequest, ProgramRequest
from .storage import (
    CURRENT_SESSION_ID,
    MEALS,
    PROGRAMS,
    get_conversation_messages,
    get_optional_session_id,
    get_session_id,
    remember_conversation_turn,
)


load_dotenv()

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
model = ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-5.5", temperature=0.5)
search_tool = TavilySearch()

app = Flask(__name__)
CORS(app)


@tool(args_schema=ProgramRequest)
def create_program(
    goal: str,
    experience_level: str,
    weeks: int,
    days_per_week: int,
    title: str | None = None,
    equipment: list[str] | None = None,
    limitations: str | None = None,
    preferences: str | None = None,
    notes: str | None = None,
) -> dict:
    """Create a personalized training program after enough user context has been collected."""
    program_request = ProgramRequest(
        goal=goal,
        experience_level=experience_level,
        weeks=weeks,
        days_per_week=days_per_week,
        title=title,
        equipment=equipment or [],
        limitations=limitations,
        preferences=preferences,
        notes=notes,
    )
    return run_program_agent(program_request, model, search_tool).model_dump()


@tool(args_schema=DietRequest)
def submit_diet_entry(
    food: list[dict[str, str]],
    meal: str,
    date: str | None = None,
    time: str | None = None,
) -> dict:
    """Submit a meal entry to the diet workflow and return estimated macros."""
    diet_request = DietRequest(food=food, date=date, time=time, meal=meal)
    return run_diet_agent(diet_request, model, search_tool).model_dump()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/meals")
def get_meals():
    session_id = get_optional_session_id(request.args)
    return jsonify({
        "session_id": session_id,
        "meals": MEALS.get(session_id, []) if session_id else [],
    })


@app.get("/programs")
def get_programs():
    session_id = get_optional_session_id(request.args)
    return jsonify({
        "session_id": session_id,
        "programs": PROGRAMS.get(session_id, []) if session_id else [],
    })


@app.get("/next-workout")
def get_next_workout():
    session_id = get_optional_session_id(request.args)
    programs = PROGRAMS.get(session_id, []) if session_id else []

    if not session_id:
        return jsonify({
            "session_id": None,
            "next_workout": None,
            "message": "No session_id provided.",
        })

    if not programs:
        return jsonify({
            "session_id": session_id,
            "next_workout": None,
            "message": "No programs stored for this session.",
        })

    latest_program = programs[-1]
    program_response = latest_program.get("response", {})
    workouts = program_response.get("workouts", [])
    next_workout = next(
        (workout for workout in workouts if not workout.get("completed", False)),
        None,
    )

    return jsonify({
        "session_id": session_id,
        "program_title": program_response.get("title"),
        "next_workout": next_workout,
        "message": None if next_workout else "All workouts in the latest program are completed.",
    })


@app.post("/chat")
def user_message():
    data = request.get_json(silent=True) or {}
    request_message = data.get("user_message", "").strip()
    session_id = get_session_id(data)

    if not request_message:
        return {"error": "empty message or request"}, 400

    chat_agent = create_agent(
        model=model,
        tools=[search_tool, submit_diet_entry, create_program],
        system_prompt=CHAT_SYSTEM_PROMPT,
    )

    try:
        previous_messages = get_conversation_messages(session_id)
        session_token = CURRENT_SESSION_ID.set(session_id)
        try:
            result = chat_agent.invoke({
                "messages": previous_messages + [HumanMessage(content=request_message)]
            })
        finally:
            CURRENT_SESSION_ID.reset(session_token)

        assistant_message = result["messages"][-1].content
        remember_conversation_turn(session_id, request_message, assistant_message)
        return jsonify({
            "message": assistant_message,
            "session_id": session_id,
        })
    except Exception as error:
        app.logger.exception("Chat request failed")
        return jsonify({"error": str(error)}), 500


@app.post("/diet")
def diet_entry():
    data = request.get_json(silent=True) or {}
    session_id = get_optional_session_id(data)

    try:
        diet_request = DietRequest.model_validate(data)
    except ValidationError as error:
        return jsonify({
            "error": "invalid diet request",
            "details": error.errors(),
        }), 400

    try:
        diet_response = run_diet_agent(diet_request, model, search_tool, session_id=session_id)
        return jsonify(diet_response.model_dump())
    except Exception as error:
        app.logger.exception("Diet request failed")
        return jsonify({"error": str(error)}), 500


@app.post("/program")
def program_entry():
    data = request.get_json(silent=True) or {}
    session_id = get_optional_session_id(data)

    try:
        program_request = ProgramRequest.model_validate(data)
    except ValidationError as error:
        return jsonify({
            "error": "invalid program request",
            "details": error.errors(),
        }), 400

    try:
        program_response = run_program_agent(
            program_request,
            model,
            search_tool,
            session_id=session_id,
        )
        return jsonify(program_response.model_dump())
    except Exception as error:
        app.logger.exception("Program request failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
