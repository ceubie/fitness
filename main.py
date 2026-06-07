from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from contextvars import ContextVar
import datetime
import json
import os
import uuid
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.messages import AIMessage, HumanMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError



load_dotenv()

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
model = ChatOpenAI(api_key=OPEN_AI_KEY, model="gpt-5.5", temperature=0.5)
search_tool = TavilySearch()
app = Flask(__name__)
CORS(app)
CONVERSATIONS = {}
MAX_CONVERSATION_MESSAGES = 30
PROGRAMS = {}
MEALS = {}
CURRENT_SESSION_ID = ContextVar("CURRENT_SESSION_ID", default=None)

class DietRequest(BaseModel):
    food: list[dict[str, str]] = Field(..., min_length=1)
    date: str | None = None
    time: str | None = None
    meal: str


class DietResponse(BaseModel):
    calories: int
    protein: int
    carbohydrates: int
    fats: int
    meal: str
    food: str


class Exercise(BaseModel):
    name: str
    sets: int = Field(..., ge=1)
    reps: str
    weight: str | None = None
    rest: str | None = None
    notes: str | None = None


class WorkoutDay(BaseModel):
    day: str
    focus: str
    exercises: list[Exercise] = Field(..., min_length=1)
    completed: bool = Field(default=False)


class ProgramRequest(BaseModel):
    goal: str
    experience_level: str
    weeks: int = Field(..., ge=1)
    days_per_week: int = Field(..., ge=1, le=7)
    title: str | None = None
    equipment: list[str] = Field(default_factory=list)
    limitations: str | None = None
    preferences: str | None = None
    notes: str | None = None


class ProgramResponse(BaseModel):
    title: str
    program: str = Field(..., min_length=1)
    weeks: int
    days_per_week: int
    split: str
    workouts: list[WorkoutDay] = Field(..., min_length=1, max_length=7)
    notes: str | None = None



DIET_SYSTEM_PROMPT = """
You are a nutrition assistant that returns meal macro estimates.
You must use the search tool to look up nutrition information for the provided foods and amounts.
Search for reliable nutrition data for each food item, including calories, protein, carbohydrates, and fats.
Calculate totals for the complete meal from the searched nutrition data.
Do not use hardcoded nutrition assumptions.
Round calories, protein, carbohydrates, and fats to whole numbers.
The food field should be a concise natural-language summary of the meal.
Return only the requested structured response shape.
"""

PROGRAM_SYSTEM_PROMPT = """
You are an expert exercise programming specialist.
You receive a complete ProgramRequest as JSON and must create the final personalized training program.
Do not ask follow-up questions in this agent. The chat agent is responsible for collecting context before calling you.
Use the search tool when useful to check current evidence-based programming guidance.

Return only the requested structured ProgramResponse shape.
The workouts field is required and must never be empty.
Create exactly days_per_week workout days.
Each workout day must include a day name, focus, and at least 3 exercises.
Each exercise must include name, sets, and reps.
Use weight only when the user provided enough information to prescribe it safely; otherwise use null and provide intensity guidance in notes.
Include rest and exercise notes when useful.
The program text should summarize progression, weekly structure, intensity, deloads if appropriate, and safety notes.
"""

CHAT_SYSTEM_PROMPT = """
You are a helpful chat assistant.
If the user asks a normal question, answer normally.
If the user says they ate, had, consumed, logged, or wants to record a meal entry, call submit_diet_entry.
If the user requests a new training program, first collect enough information to create one safely.
Program intake must be low-friction and sequential.
Ask exactly one program follow-up question per assistant turn.
Never dump a numbered list, checklist, or multiple program questions at once.
When information is missing, ask for the next missing field only, in this order:
1. goal
2. available weeks
3. days per week
4. experience level and relevant skill level
5. equipment
6. limitations or injuries
7. preferences
If the user's latest message answers more than one field, use all provided information silently and ask only the next missing field.
If the prior assistant message asked a program follow-up question, treat the user's latest message as the answer to that question and continue to the next missing field.
Ask clarifying questions instead of calling create_program when goal, experience level, available weeks, days per week, equipment, limitations/injuries, or preferences are unclear.
Call create_program only after you have enough information.
When calling submit_diet_entry, format the meal into this request shape:
{"food": [{"food name": "amount"}], "date": "YYYY-MM-DD", "time": "HH:MM", "meal": "breakfast|lunch|dinner|snack|meal"}
Use date and time only if the user provides them; otherwise omit them.
Infer meal from the user's words when possible; if unclear, use "meal".
After submit_diet_entry returns, summarize the recorded meal and macros for the user.
When calling create_program, format the program request into this shape:
{"goal": "strength|hypertrophy|endurance|general fitness|...", "experience_level": "beginner|intermediate|advanced", "weeks": 8, "days_per_week": 4, "title": "optional title", "equipment": ["barbell", "dumbbells"], "limitations": "injuries or constraints", "preferences": "preferred split/exercises/style", "notes": "other relevant context"}
After create_program returns, give a concise breakdown of the created program.
For program-intake replies before create_program, keep the response brief: one sentence of acknowledgement at most, then one question.
If current or external information is needed for a normal question, use the search tool.
"""


def default_diet_timestamp(diet_request: DietRequest) -> DietRequest:
    now = datetime.datetime.now()
    return diet_request.model_copy(update={
        "date": diet_request.date or now.date().isoformat(),
        "time": diet_request.time or now.strftime("%H:%M"),
    })


def get_session_id(data: dict) -> str:
    session_id = get_optional_session_id(data)

    if session_id:
        return session_id

    return str(uuid.uuid4())


def get_optional_session_id(data: dict) -> str | None:
    session_id = data.get("session_id")

    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()

    return None


def get_conversation_messages(session_id: str) -> list:
    return CONVERSATIONS.setdefault(session_id, [])


def remember_conversation_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    messages = get_conversation_messages(session_id)
    messages.extend([
        HumanMessage(content=user_message),
        AIMessage(content=assistant_message),
    ])
    CONVERSATIONS[session_id] = messages[-MAX_CONVERSATION_MESSAGES:]


def get_current_session_id() -> str | None:
    return CURRENT_SESSION_ID.get()


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


def run_diet_agent(diet_request: DietRequest, session_id: str | None = None) -> DietResponse:
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

    res = DietResponse.model_validate(diet_response)
    remember_meal(session_id or get_current_session_id(), diet_request, res)

    return res

def run_program_agent(program_request: ProgramRequest, session_id: str | None = None) -> ProgramResponse:
    program_agent = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=PROGRAM_SYSTEM_PROMPT,
        response_format=ProgramResponse
    )
    result = program_agent.invoke({
        "messages": [
            HumanMessage(content=json.dumps(program_request.model_dump()))
        ]
        })
    
    program_response = result.get("structured_response")

    if program_response is None:
        raise ValueError("program agent did not return structured output")

    res = ProgramResponse.model_validate(program_response)
    remember_program(session_id or get_current_session_id(), program_request, res)
    return res


# def build_fallback_program(program_request: ProgramRequest) -> ProgramResponse:
#     split_options = {
#         1: ["Full Body"],
#         2: ["Upper Body", "Lower Body"],
#         3: ["Full Body A", "Full Body B", "Full Body C"],
#         4: ["Upper Strength", "Lower Strength", "Upper Volume", "Lower Volume"],
#         5: ["Upper", "Lower", "Push", "Pull", "Legs"],
#         6: ["Push", "Pull", "Legs", "Upper", "Lower", "Conditioning"],
#         7: ["Push", "Pull", "Legs", "Upper", "Lower", "Conditioning", "Mobility"],
#     }
#     focus_list = split_options[program_request.days_per_week]
#     title = program_request.title or f"{program_request.weeks}-Week {program_request.goal.title()} Program"
#     workouts = []

#     exercise_templates = {
#         "upper": ["Bench Press", "Row", "Overhead Press", "Pulldown", "Lateral Raise"],
#         "lower": ["Squat", "Romanian Deadlift", "Split Squat", "Leg Curl", "Calf Raise"],
#         "push": ["Bench Press", "Overhead Press", "Incline Press", "Triceps Pressdown", "Lateral Raise"],
#         "pull": ["Deadlift", "Pull-Up or Pulldown", "Row", "Face Pull", "Curl"],
#         "legs": ["Squat", "Romanian Deadlift", "Lunge", "Leg Curl", "Calf Raise"],
#         "conditioning": ["Zone 2 Cardio", "Intervals", "Loaded Carry"],
#         "mobility": ["Hip Mobility", "Thoracic Rotation", "Hamstring Mobility"],
#         "full": ["Squat", "Bench Press", "Row", "Romanian Deadlift", "Core Work"],
#     }

#     for index, focus in enumerate(focus_list, start=1):
#         focus_key = focus.lower().split()[0]
#         template = exercise_templates.get(focus_key, exercise_templates["full"])
#         exercises = [
#             Exercise(
#                 name=exercise_name,
#                 sets=3,
#                 reps="6-10" if program_request.goal.lower() in {"strength", "hypertrophy"} else "8-12",
#                 rest="2-3 min" if index <= 2 else "60-90 sec",
#                 notes="Start conservatively and add load or reps when all sets are completed with good form.",
#             )
#             for exercise_name in template
#         ]
#         workouts.append(WorkoutDay(
#             day=f"Day {index}",
#             focus=focus,
#             exercises=exercises,
#         ))

#     return ProgramResponse(
#         title=title,
#         program=(
#             f"{title}: train {program_request.days_per_week} days per week for "
#             f"{program_request.weeks} weeks. Progress by adding 1-2 reps or small load increases "
#             "when technique remains solid. Keep 1-3 reps in reserve on most work sets."
#         ),
#         weeks=program_request.weeks,
#         days_per_week=program_request.days_per_week,
#         split=", ".join(focus_list),
#         workouts=workouts,
#         notes=program_request.notes or "Fallback generated because the model returned invalid structured output.",
#     )

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
    notes: str | None = None
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

    try:
        return run_program_agent(program_request).model_dump()
    except Exception as error:
        app.logger.exception("Program agent returned invalid output; using fallback program")
        # fallback_program = build_fallback_program(program_request).model_dump()
        # fallback_program["warning"] = f"Used fallback program because the program agent failed: {error}"
        # return fallback_program

@tool(args_schema=DietRequest)
def submit_diet_entry(
    food: list[dict[str, str]],
    meal: str,
    date: str | None = None,
    time: str | None = None,
) -> dict:
    """Submit a meal entry to the diet workflow and return estimated macros."""
    diet_request = DietRequest(food=food, date=date, time=time, meal=meal)
    return run_diet_agent(diet_request).model_dump()

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/health')
def health():
    return jsonify({"status": "ok"})

@app.get('/meals')
def get_meals():
    session_id = get_optional_session_id(request.args)
    return jsonify({
        "session_id": session_id,
        "meals": MEALS.get(session_id, []) if session_id else [],
    })

@app.get('/programs')
def get_programs():
    session_id = get_optional_session_id(request.args)
    return jsonify({
        "session_id": session_id,
        "programs": PROGRAMS.get(session_id, []) if session_id else [],
    })

@app.get('/next-workout')
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

@app.post('/chat')
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
    
@app.post('/diet')
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
        diet_response = run_diet_agent(diet_request, session_id=session_id)
        return jsonify(diet_response.model_dump())
    except Exception as error:
        app.logger.exception("Diet request failed")
        return jsonify({"error": str(error)}), 500
    
@app.post('/program')
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
        program_response = run_program_agent(program_request, session_id=session_id)
        return jsonify(program_response.model_dump())
    except Exception as error:
        app.logger.exception("Program agent returned invalid output; using fallback program")
        # fallback_program = build_fallback_program(program_request).model_dump()
        # fallback_program["warning"] = f"Used fallback program because the program agent failed: {error}"
        # return jsonify(fallback_program), 200
    


if __name__ == "__main__":
    app.run(debug=True, port=5000)
