from .load_model import model
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

top_countries=['Other','Canada','USA','India','Australia','UK','Germany','France','Mexico','Turkey']

class Data(BaseModel):
    Age:int=Field(...,ge=10,le=100)
    Gender:Literal['Male','Female']
    Country:str
    Academic_Level:Literal['Undergraduate','Graduate','High School']
    Most_Used_Platform:Literal['Facebook','LinkedIn','Instagram','Snapchat','Twitter','YouTube','TikTok','LINE','KakaoTalk','VKontakte','WhatsApp','WeChat']
    Purpose_Of_Use:Literal['Networking','Education','Entertainment','News']
    Avg_Daily_Usage_Hours:int=Field(...,ge=0,le=24)
    Daily_Unlocks:int=Field(...,ge=0,le=24)
    Study_Hours:int=Field(...,ge=0,le=24)
    Physical_Activity_Hours:int=Field(...,ge=0,le=24)
    Sleep_Hours_Per_Night:int=Field(...,ge=0,le=24)
    Stress_Level:Literal['Medium','Low','Very High','High']

@app.get("/")
def greet():
    return "Welcome User"

class PredictionResponse(BaseModel):
    prediction:float

@app.post("/predict",response_model=PredictionResponse)
def predict(data:Data):
    if data.Country not in top_countries:
        country="Other"
    else:
        country=data.Country
    input_rows=pd.DataFrame([{
    "Age":data.Age,
    "Gender":data.Gender,
    "Country":country,
    "Academic_Level":data.Academic_Level,
    "Most_Used_Platform":data.Most_Used_Platform,
    "Purpose_Of_Use":data.Purpose_Of_Use,
    "Avg_Daily_Usage_Hours":data.Avg_Daily_Usage_Hours,
    "Daily_Unlocks":data.Daily_Unlocks,
    "Study_Hours":data.Study_Hours,
    "Physical_Activity_Hours":data.Physical_Activity_Hours,
    "Sleep_Hours_Per_Night":data.Sleep_Hours_Per_Night,
    "Stress_Level":data.Stress_Level
    }])
    prediction=model.predict(input_rows)[0]
    return PredictionResponse(prediction=round(float(prediction), 2))




