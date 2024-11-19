import pandas as pd
import joblib
import numpy as np
import typing
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

with open('LRmodel.pkl', 'rb') as file:
    model = joblib.load(file)

class ModelRequestData(BaseModel):
    lat: float
    lon: float
    total_square: float
    rooms: int
    floor: int
    

class Result(BaseModel):
    result: float

@app.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    result = 'Я жив'
    return result

@app.post(path="/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.get(path="/predict_get", response_model=Result)
def predict_get(
    lat: float, 
    lon: float, 
    total_square: float, 
    rooms: int, 
    floor: int
):
    input_data = {
        "lat": lat,
        "lon": lon,
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor
    }
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)
