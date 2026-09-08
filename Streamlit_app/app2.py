# This version upgrades your app with:
# - Better layout (insight graphs ABOVE prediction)
# - General lifestyle remarks based on input
# - More relationships from input
# - Cleaner visuals with friendly titles
# - Distinct sections with headers

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lifestyle Disease Predictor", layout="wide")

# ---------------------- Load Model Artifacts ----------------------
@st.cache_resource

def load_artifacts():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_artifacts()

TARGET_LABELS = [
    'Heart_Disease_Yes', 'Skin_Cancer_Yes', 'Other_Cancer_Yes',
    'Depression_Yes', 'Diabetes_Yes', 'Arthritis_Yes']

TARGET_FRIENDLY = {
    'Heart_Disease_Yes':  'Heart Disease',
    'Skin_Cancer_Yes':    'Skin Cancer',
    'Other_Cancer_Yes':   'Other Cancer',
    'Depression_Yes':     'Depression',
    'Diabetes_Yes':       'Diabetes',
    'Arthritis_Yes':      'Arthritis'
}

ADVICE = {
    'Heart_Disease_Yes':  "Manage stress, exercise often, avoid smoking & fatty foods.",
    'Skin_Cancer_Yes':    "Protect your skin, wear sunscreen, and consult dermatologists.",
    'Other_Cancer_Yes':   "Get screening tests regularly and live a low-risk lifestyle.",
    'Depression_Yes':     "Consider speaking to a mental health professional.",
    'Diabetes_Yes':       "Monitor sugar intake, get regular exercise and check-ups.",
    'Arthritis_Yes':      "Exercise regularly, manage weight and consider physiotherapy."
}

# ---------------------- Helper Functions ----------------------
def build_feature_row_from_inputs(raw_inputs: dict, feature_cols: list) -> pd.DataFrame:
    row = pd.Series(0, index=feature_cols, dtype=float)
    numeric_keys = [
        'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI',
        'Alcohol_Consumption', 'Fruit_Consumption',
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    for k in numeric_keys:
        if k in row.index and k in raw_inputs:
            row[k] = raw_inputs[k]

    def set_onehot(prefix, chosen_value):
        col = f"{prefix}_{chosen_value}"
        if col in row.index:
            row[col] = 1.0

    set_onehot("General_Health", raw_inputs["General_Health"])
    set_onehot("Checkup", raw_inputs["Checkup"])
    set_onehot("Exercise", raw_inputs["Exercise"])
    set_onehot("Sex", raw_inputs["Sex"])
    set_onehot("Smoking_History", raw_inputs["Smoking_History"])

    return pd.DataFrame([row.values], columns=row.index)

# ---------------------- UI ----------------------
st.title("🌿 Lifestyle Disease Predictor")
st.write("""
This tool uses machine learning to predict possible lifestyle-related diseases.
**Please consult your doctor for medical advice.**
""")

# Sidebar Metadata
with st.sidebar:
    st.header("📂 Model Info")
    st.write(f"Trained on **{len(FEATURE_COLS)}** features")
    st.write("Predicted conditions:")
    for t in TARGET_LABELS:
        st.markdown(f"- {TARGET_FRIENDLY[t]}")

# Tabs
prediction_tab, dashboard_tab = st.tabs(["🩺 Prediction", "📊 Dashboard"])

# ---------------------- PREDICTION TAB ----------------------
with prediction_tab:
    st.markdown("### 👤 Enter Your Information")

    age_map = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
               "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
               "70-74": 11, "75-79": 12, "80+": 13}
    
    age_choice = st.selectbox("Age Category", list(age_map.keys()))
    sex = st.selectbox("Sex", ["Male", "Female"])
    general = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
    checkup = st.selectbox("Last Checkup", ["Within the past year", "Within the past 2 years",
                                             "Within the past 5 years", "5 or more years ago", "Never"])
    exercise = st.selectbox("Exercise Regularly?", ["Yes", "No"])
    smoking = st.selectbox("Smoking History", ["Yes", "No"])

    c1, c2 = st.columns(2)
    with c1:
        height = st.slider("Height (cm)", 120.0, 220.0, 170.0)
        weight = st.slider("Weight (kg)", 35.0, 200.0, 75.0)
        alcohol = st.slider("Drinks/week", 0, 30, 2)
    with c2:
        fruit = st.slider("Fruit servings/month", 0, 60, 20)
        green = st.slider("Green veggies/month", 0, 60, 15)
        fried = st.slider("Fried potato servings/month", 0, 30, 5)

    bmi = round(weight / ((height/100)**2), 2)
    st.markdown(f"**Calculated BMI:** `{bmi}`")

    raw_input = {
        "Age_Category": age_map[age_choice],
        "Height_(cm)": height,
        "Weight_(kg)": weight,
        "BMI": bmi,
        "Alcohol_Consumption": alcohol,
        "Fruit_Consumption": fruit,
        "Green_Vegetables_Consumption": green,
        "FriedPotato_Consumption": fried,
        "General_Health": general,
        "Checkup": checkup,
        "Exercise": exercise,
        "Sex": sex,
        "Smoking_History": smoking
    }

    if st.button("🧮 Predict Risk"):
        X_user = build_feature_row_from_inputs(raw_input, FEATURE_COLS)
        X_scaled = scaler.transform(X_user)
        preds = model.predict(X_scaled)[0]

        results = dict(zip(TARGET_LABELS, preds))
        st.session_state["last_input"] = raw_input
        st.session_state["last_pred"] = results

        st.markdown("---")
        st.subheader("🩻 Prediction Results")
        positives = []

        for label, pred in results.items():
            name = TARGET_FRIENDLY[label]
            if pred == 1:
                st.error(f"⚠️ {name}: High Risk")
                st.markdown(f"> 💡 {ADVICE[label]}")
                positives.append(name)
            else:
                st.success(f"✅ {name}: Low Risk")

        st.markdown("---")
        st.subheader("🩺 General Lifestyle Advice")
        if bmi > 30:
            st.warning("Your BMI indicates obesity. Consider weight management support.")
        elif bmi < 18:
            st.warning("Your BMI is below normal range. Consider nutritional advice.")
        else:
            st.success("Your BMI is within the healthy range.")

        if exercise == "No":
            st.info("Engaging in regular physical activity can greatly reduce risk.")
        if fruit < 10:
            st.info("Increase fruit intake for antioxidant benefits.")

# ---------------------- DASHBOARD TAB ----------------------
with dashboard_tab:
    st.header("📊 Visual Insights Based on Your Data")

    if "last_input" in st.session_state:
        data = st.session_state["last_input"]
        pred = st.session_state["last_pred"]

        # DISTRIBUTIONS — lifestyle first
        st.subheader("🍎 Lifestyle Overview")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x=["Fruit", "Veggies", "Fried"],
                        y=[data["Fruit_Consumption"], data["Green_Vegetables_Consumption"], data["FriedPotato_Consumption"]],
                        ax=ax1, palette="summer")
            ax1.set_title("Diet Pattern (Monthly Servings)")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x=["Alcohol"], y=[data["Alcohol_Consumption"]], palette=["#E74C3C"], ax=ax2)
            ax2.set_title("Alcohol Consumption")
            st.pyplot(fig2)

        st.subheader("📈 Your Risk Score Summary")
        fig3, ax3 = plt.subplots()
        sns.barplot(x=[TARGET_FRIENDLY[k] for k in pred.keys()], y=[v for v in pred.values()],
                    ax=ax3, palette="coolwarm")
        ax3.set_title("Predicted Risk Flags")
        ax3.set_ylabel("1 = High Risk")
        st.pyplot(fig3)
    else:
        st.info("❗ Run a prediction to see your dashboard.")
