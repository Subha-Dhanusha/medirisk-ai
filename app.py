# app.py
import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediRisk AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediRisk AI — Readmission Risk Predictor")
st.markdown("Enter patient details to predict 30-day hospital readmission risk.")

# --- Sidebar: Patient Form ---
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 1, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 1)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 132, 40)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 10)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 5)

st.sidebar.subheader("Visit History")
number_outpatient = st.sidebar.number_input("Outpatient Visits", 0, 50, 0)
number_emergency = st.sidebar.number_input("Emergency Visits", 0, 50, 0)
number_inpatient = st.sidebar.number_input("Prior Inpatient Visits", 0, 50, 0)

st.sidebar.subheader("Diagnoses")
diag_options = {
    "Other (0)": 0, "Circulatory (1)": 1, "Respiratory (2)": 2,
    "Digestive (3)": 3, "Diabetes (4)": 4, "Injury (5)": 5,
    "Musculoskeletal (6)": 6, "Genitourinary (7)": 7, "Neoplasms (8)": 8
}
diag_1 = st.sidebar.selectbox("Primary Diagnosis", list(diag_options.keys()), index=4)
diag_2 = st.sidebar.selectbox("Secondary Diagnosis", list(diag_options.keys()), index=1)
diag_3 = st.sidebar.selectbox("Tertiary Diagnosis", list(diag_options.keys()), index=0)

st.sidebar.subheader("Diabetes Management")
diabetesMed = st.sidebar.selectbox("On Diabetes Medication?", ["No", "Yes"])
change = st.sidebar.selectbox("Medication Changed?", ["No", "Yes"])
insulin_opts = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
insulin = st.sidebar.selectbox("Insulin", list(insulin_opts.keys()), index=1)

st.sidebar.subheader("Race")
race = st.sidebar.selectbox("Race", ["African American", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"])

# --- Predict Button ---
if st.sidebar.button("🔍 Predict Risk", use_container_width=True):
    payload = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "time_in_hospital": time_in_hospital,
        "num_procedures": num_procedures,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "diag_1": diag_options[diag_1],
        "diag_2": diag_options[diag_2],
        "diag_3": diag_options[diag_3],
        "diabetesMed": 1 if diabetesMed == "Yes" else 0,
        "change": 1 if change == "Yes" else 0,
        "insulin": insulin_opts[insulin],
        "metformin": 0, "glipizide": 0, "glyburide": 0,
        "pioglitazone": 0, "rosiglitazone": 0, "glimepiride": 0,
        "race_Asian": race == "Asian",
        "race_Caucasian": race == "Caucasian",
        "race_Hispanic": race == "Hispanic",
        "race_Other": race == "Other",
        "race_Unknown": race == "Unknown",
    }

    with st.spinner("Analyzing patient data..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            # --- Results ---
            col1, col2, col3 = st.columns(3)

            risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
            color = risk_color.get(result["risk_level"], "gray")

            col1.metric("Risk Score", f"{result['risk_score']}%")
            col2.metric("Risk Level", result["risk_level"])
            col3.metric("Readmission Probability", f"{result['readmission_probability']*100:.1f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["risk_score"],
                title={"text": "Readmission Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 20], "color": "#d4edda"},
                        {"range": [20, 40], "color": "#fff3cd"},
                        {"range": [40, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": result["risk_score"]
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Recommendation
            st.info(f"📋 **Recommendation:** {result['recommendation']}")

            # Top risk factors
            st.subheader("Top Risk Factors")
            features = [f["feature"] for f in result["top_risk_factors"]]
            importances = [f["importance"] for f in result["top_risk_factors"]]

            fig2 = go.Figure(go.Bar(
                x=importances, y=features,
                orientation="h",
                marker_color="#4e79a7"
            ))
            fig2.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
else:
    st.info("👈 Fill in patient details in the sidebar and click **Predict Risk**")
    st.markdown("""
    ### How it works
    1. Enter patient demographics and clinical data in the sidebar
    2. Click **Predict Risk**
    3. The AI model returns a readmission risk score
    4. Use the recommendation to guide care planning
    """)