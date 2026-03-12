from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from features import prepare_features
import dill
import pandas as pd
import time


# Инструкция по запуску:
# 1. Установить зависимости: pip install -r requirements.txt
# 2. Запустить сервер: uvicorn main:app --reload
# 3. Открыть документацию: http://127.0.0.1:8000/docs

app = FastAPI()
with open("model/sber_auto_model.pkl", 'rb') as file:
    artifacts = dill.load(file)

model = artifacts['model']

class SessionData(BaseModel):
    session_id: str
    client_id: str
    visit_number: int
    visit_date: str
    visit_time: str
    utm_source: Optional[str] = None
    utm_medium: str
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: str
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_browser: str
    device_screen_resolution: str
    geo_country: str
    geo_city: str



@app.get('/status')
def status():
    return 'I`m OK'

@app.post('/predict')
def predict(session_data: SessionData):
    start_time = time.time()

    df = pd.DataFrame([session_data.dict()])
    df = prepare_features(df, artifacts)
    prediction = model.predict(df)[0]
    print(f'{time.time() - start_time:.3f}c')
    return {'prediction': int(prediction)}

#"Среднее время ответа API составляет ~28мс, что значительно меньше требуемых 3 секунд."

