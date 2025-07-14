from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("eligibility_model.joblib")

# Créer l’application
app = FastAPI()

# Définir les champs attendus (à adapter à ton modèle)
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    NAME_TYPE_SUITE: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float


@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict_proba(df)[0][1]
    return {"score_eligibilite": round(prediction, 3)}
