import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------
# Load artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")  # saved from your notebook
    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_artifacts()

# Targets – in the order you trained / evaluated
TARGET_LABELS = [
    'Heart_Disease_Yes',
    'Skin_Cancer_Yes',
    'Other_Cancer_Yes',
    'Depression_Yes',
    'Diabetes_Yes',
    'Arthritis_Yes'
]

# (Optional) friendly names for UI output
TARGET_FRIENDLY = {
    'Heart_Disease_Yes':  'Heart Disease',
    'Skin_Cancer_Yes':    'Skin Cancer',
    'Other_Cancer_Yes':   'Other Cancer',
    'Depression_Yes':     'Depression',
    'Diabetes_Yes':       'Diabetes',
    'Arthritis_Yes':      'Arthritis'
}

# Advice text (very simple placeholders — customize!)
ADVICE = {
    'Heart_Disease_Yes':  "Consider seeing a cardiologist. Manage BP, cholesterol, and exercise regularly.",
    'Skin_Cancer_Yes':    "See a dermatologist for a skin check. Use sunscreen and monitor lesions.",
    'Other_Cancer_Yes':   "Consult your physician for further screening and diagnostics.",
    'Depression_Yes':     "Reach out to a mental-health professional. Therapy and support groups help.",
    'Diabetes_Yes':       "Monitor glucose, diet, and physical activity. Consult an endocrinologist.",
    'Arthritis_Yes':      "Manage weight, stay active, and consult a rheumatologist if pain persists."
}

# ---------------------------
# Helper: one-hot encode a single user input dict to the model's FEATURE_COLS
# ---------------------------
def build_feature_row_from_inputs(raw_inputs: dict, feature_cols: list) -> pd.DataFrame:
    """
    raw_inputs: dict holding raw fields (numeric + categorical selections)
    feature_cols: exact training feature names (one-hot included)
    Returns a single-row DataFrame with the exact columns the model expects.
    """
    # 1) Start with all-zero vector
    row = pd.Series(0, index=feature_cols, dtype=float)

    # 2) Fill numeric columns directly (if they exist in feature_cols)
    numeric_keys = [
        'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI',
        'Alcohol_Consumption', 'Fruit_Consumption',
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    for k in numeric_keys:
        if k in row.index and k in raw_inputs:
            row[k] = raw_inputs[k]

    # 3) One-hot categorical columns
    #    NOTE: Keys below must match how they appeared after pd.get_dummies in your notebook
    #    We assume you encoded with drop_first=False, so both categories exist as columns.
    def set_onehot(prefix, chosen_value):
        # Ex: prefix='General_Health', chosen='Good' -> column 'General_Health_Good'
        col = f"{prefix}_{chosen_value}"
        if col in row.index:
            row[col] = 1.0

    set_onehot("General_Health", raw_inputs["General_Health"])
    set_onehot("Checkup",        raw_inputs["Checkup"])
    set_onehot("Exercise",       raw_inputs["Exercise"])
    set_onehot("Sex",            raw_inputs["Sex"])
    set_onehot("Smoking_History",raw_inputs["Smoking_History"])

    # Return a DF with single row
    return pd.DataFrame([row.values], columns=row.index)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Lifestyle Disease Prediction", page_icon="🩺", layout="wide")

st.title("🩺 Lifestyle Disease Prediction ")
st.write(
    """
    Enter your health and lifestyle information below to get multi-disease risk predictions
    using the trained **XGBoost** model.  
    *(This is **not** a medical diagnosis — consult a clinician for medical advice.)*
    """
)

with st.sidebar:
    st.header("📦 Model & Feature Info")
    st.write(f"**Features used:** {len(FEATURE_COLS)}")
    st.write(f"**Targets:** {', '.join([TARGET_FRIENDLY[t] for t in TARGET_LABELS])}")

st.markdown("### 🔧 Provide your details")

# ---------------------------
# Collect user inputs
# ---------------------------

# Age category stored as numeric (your notebook mapped 13 groups to 1..13 or similar)
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
    "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13
}
age_choice = st.selectbox("Age Category", list(age_map.keys()))
age_numeric = age_map[age_choice]

sex_choice = st.selectbox("Sex", ["Male", "Female"])
general_health_choice = st.selectbox("General Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
checkup_choice = st.selectbox(
    "Last Routine Checkup",
    ["Within the past year", "Within the past 2 years", "Within the past 5 years",
     "5 or more years ago", "Never"]
)
exercise_choice = st.selectbox("Do you exercise?", ["Yes", "No"])
smoking_choice = st.selectbox("Smoking history", ["Yes", "No"])

col1, col2 = st.columns(2)
with col1:
    height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
    weight = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=75.0, step=0.5)
    alcohol = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=30, value=1, step=1)
    fried = st.number_input("Fried Potato Consumption (per month)", min_value=0, max_value=30, value=4, step=1)
with col2:
    fruit = st.number_input("Fruit Consumption (servings/month)", min_value=0, max_value=60, value=30, step=1)
    greens = st.number_input("Green Vegetables Consumption (servings/month)", min_value=0, max_value=60, value=12, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(round(weight / ((height/100)**2), 2)))
    st.caption("BMI auto-filled from height & weight; adjust if needed.")

# Build a raw input dict
raw_input = {
    "Age_Category": age_numeric,
    "Height_(cm)": height,
    "Weight_(kg)": weight,
    "BMI": bmi,
    "Alcohol_Consumption": alcohol,
    "Fruit_Consumption": fruit,
    "Green_Vegetables_Consumption": greens,
    "FriedPotato_Consumption": fried,
    "General_Health": general_health_choice,
    "Checkup": checkup_choice,
    "Exercise": exercise_choice,
    "Sex": sex_choice,
    "Smoking_History": smoking_choice
}

if st.button("🔮 Predict"):
    # 1) Build encoded feature row
    X_user = build_feature_row_from_inputs(raw_input, FEATURE_COLS)

    # 2) Scale
    X_user_scaled = scaler.transform(X_user)

    # 3) Predict
    preds = model.predict(X_user_scaled)[0]

    # 4) Present results nicely
    st.subheader("🧾 Prediction")
    results = dict(zip(TARGET_LABELS, preds))

    positive_conditions = []
    for k, v in results.items():
        nice = TARGET_FRIENDLY[k]
        if v == 1:
            st.error(f"**{nice}: High risk detected** ⚠️")
            st.write(f"**What to do:** {ADVICE.get(k, 'Consult a clinician.')}")
            positive_conditions.append(nice)
        else:
            st.success(f"{nice}: Low risk")

    if len(positive_conditions) == 0:
        st.info("✅ No high-risk conditions detected by the model.")
    else:
        st.markdown("### 📌 Summary")
        st.write("High-risk flags:", ", ".join(positive_conditions))

    # Optional: show raw vector (debug)
    with st.expander("🔍 See encoded feature vector (debug)"):
        st.dataframe(pd.DataFrame(X_user, columns=FEATURE_COLS).T, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + XGBoost. Not medical advice.")
