from pydantic import BaseModel, Field


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


class SetEntry(BaseModel):
    number: int
    reps: int
    load: int
    unit: str
    rpe: int = Field(ge=1, le=10)
    notes: str


class Movement(BaseModel):
    name: str
    sets: list[SetEntry]
    notes: str


class WorkoutEntry(BaseModel):
    program: str
    week: int
    day_number: int
    exercises: list[Movement]
    notes: str
    date: str
    time: str
