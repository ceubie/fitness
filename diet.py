import datetime
request = {
    "food": [
        {"chicken breast": "150g"},
        {"rice": "175g"},
        {"brocolli": "6oz"}
    ],
    "date": datetime.date,
    "time": datetime.time,
    "meal": "lunch"
}

response = {
    "calories": 400,
    "protein": 55,
    "carbohydrates": 60,
    "fats": 10,
    "meal": "lunch",
    "food": "Chicken and rice with brocolli. "
}

xample = [{
    "monday": 
        {
            "backsquat": 
            {
                "reps": 3,
                "sets": 3,
                "weight": 100
            },
            "deadlift": 
            {
                "reps": 3,
                "sets": 3,
                "weight": 100
            },
        },
    "tuesday": {},
    }]