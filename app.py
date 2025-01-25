from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("model.joblib")

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    input_data = np.array([[request.feature1, request.feature2, request.feature3, request.feature4]])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}