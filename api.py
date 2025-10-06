from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI()

class PredictRequest(BaseModel):
    model_name: str
    data: list  # list of records (dicts)

@app.post('/predict')
def predict(req: PredictRequest):
    model_path = os.path.join('models', req.model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail='Model not found')
    model = joblib.load(model_path)
    try:
        df = pd.DataFrame(req.data)
        preds = model.predict(df)
        return {'predictions': preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
