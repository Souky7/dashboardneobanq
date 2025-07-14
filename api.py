from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle entraîné
model = joblib.load("eligibility_model.joblib")
expected_features = model.feature_names_in_

# Création de l'API FastAPI
app = FastAPI(title="API Score Éligibilité", description="Renvoie le score d’éligibilité d’un client")

# Exemple simplifié de structure d’entrée (tu peux en ajouter selon besoin)
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float = 0
    AMT_CREDIT: float = 0
    AMT_ANNUITY: float = 0
    AMT_GOODS_PRICE: float = 0
    DAYS_BIRTH: int = 0
    DAYS_EMPLOYED: int = 0
    CODE_GENDER: str = "M"
    NAME_CONTRACT_TYPE: str = "Cash loans"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"
    NAME_FAMILY_STATUS: str = "Single / not married"
    NAME_HOUSING_TYPE: str = "House / apartment"
    NAME_INCOME_TYPE: str = "Working"
    FLAG_OWN_REALTY: str = "N"
    FLAG_OWN_CAR: str = "N"
    OCCUPATION_TYPE: str = "Laborers"
    CNT_FAM_MEMBERS: float = 1.0
    DAYS_REGISTRATION: float = 0
    DAYS_ID_PUBLISH: int = 0
    NAME_TYPE_SUITE: str = "Unaccompanied"

@app.post("/predict")
def predict(data: ClientData):
    try:
        # Convertir les données reçues en DataFrame
        df = pd.DataFrame([data.dict()])

        # Créer un DataFrame avec les colonnes attendues, remplir les absentes avec 0 ou NaN
        df_full = pd.DataFrame(columns=expected_features)
        df_full = df_full.append(df, ignore_index=True)
        df_full = df_full.fillna(0)  # ou choisir une autre stratégie si nécessaire

        # S'assurer que les colonnes sont dans le bon ordre
        df_full = df_full[expected_features]

        # Prédiction
        prediction = model.predict_proba(df_full)[0][1]

        return {"score_eligibilite": round(float(prediction), 3)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
