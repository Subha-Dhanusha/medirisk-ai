# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class PatientInput(BaseModel):
    # Demographics
    age: int = Field(default=65, ge=1, le=100)
    gender: int = Field(default=0, description="0=Female, 1=Male")

    # Hospital stay
    time_in_hospital: int = Field(default=4, ge=1, le=14)
    num_procedures: int = Field(default=1, ge=0, le=6)
    num_lab_procedures: int = Field(default=40, ge=1, le=132)
    num_medications: int = Field(default=10, ge=1, le=81)
    number_diagnoses: int = Field(default=5, ge=1, le=16)

    # Visit history
    number_outpatient: int = Field(default=0, ge=0)
    number_emergency: int = Field(default=0, ge=0)
    number_inpatient: int = Field(default=0, ge=0)

    # Diagnoses (mapped categories: 0=Other, 1=Circulatory, 2=Respiratory,
    #            3=Digestive, 4=Diabetes, 5=Injury, 6=Musculoskeletal,
    #            7=Genitourinary, 8=Neoplasms)
    diag_1: int = Field(default=4, ge=0, le=18)
    diag_2: int = Field(default=1, ge=0, le=18)
    diag_3: int = Field(default=0, ge=0, le=18)

    # Diabetes management
    diabetesMed: int = Field(default=1, description="0=No, 1=Yes")
    change: int = Field(default=0, description="0=No change, 1=Changed")
    insulin: int = Field(default=1, description="0=No, 1=Steady, 2=Up, 3=Down")

    # Medications (0=No, 1=Steady, 2=Up, 3=Down)
    metformin: int = Field(default=0, ge=0, le=3)
    glipizide: int = Field(default=0, ge=0, le=3)
    glyburide: int = Field(default=0, ge=0, le=3)
    pioglitazone: int = Field(default=0, ge=0, le=3)
    rosiglitazone: int = Field(default=0, ge=0, le=3)
    glimepiride: int = Field(default=0, ge=0, le=3)

    # Race (one-hot, drop_first=True so African American is baseline)
    race_Asian: bool = False
    race_Caucasian: bool = False
    race_Hispanic: bool = False
    race_Other: bool = False
    race_Unknown: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "age": 72,
                "gender": 0,
                "time_in_hospital": 5,
                "num_procedures": 2,
                "num_lab_procedures": 45,
                "num_medications": 12,
                "number_diagnoses": 7,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 2,
                "diag_1": 4,
                "diag_2": 1,
                "diag_3": 3,
                "diabetesMed": 1,
                "change": 1,
                "insulin": 2,
                "metformin": 1,
                "glipizide": 0,
                "glyburide": 0,
                "pioglitazone": 0,
                "rosiglitazone": 0,
                "glimepiride": 0,
                "race_Asian": False,
                "race_Caucasian": True,
                "race_Hispanic": False,
                "race_Other": False,
                "race_Unknown": False
            }
        }


class PredictionOutput(BaseModel):
    risk_score: float
    risk_level: str
    readmission_probability: float
    recommendation: str
    top_risk_factors: list