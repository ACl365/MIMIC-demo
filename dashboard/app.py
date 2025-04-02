import requests
import streamlit as st

# --- Configuration ---
API_URL = (
    "http://localhost:8001/predict"  # URL of the running FastAPI container on port 8001
)

# --- Page Setup ---
st.set_page_config(page_title="MIMIC Readmission Predictor", layout="wide")
st.title("🏥 MIMIC Readmission Risk Predictor")
st.write(
    """
Enter patient details below to predict the probability of 30-day hospital readmission.
This dashboard uses the features expected by the loaded `readmission_model.pkl`.
"""
)

# --- Input Features ---
st.header("Patient Features")

# --- Feature Definitions ---
# These lists define the options for select boxes and map them to the feature names
gender_options = {"Female": "gender_f", "Male": "gender_m", "Unknown": "gender_nan"}
admission_type_options = {
    "Ambulatory Observation": "admission_type_ambulatory observation",
    "Direct Emergency": "admission_type_direct emer.",
    "Direct Observation": "admission_type_direct observation",
    "Elective": "admission_type_elective",
    "Emergency": "admission_type_emergency",
    "EU Observation": "admission_type_eu observation",
    "EW Emergency": "admission_type_ew emer.",
    "Observation Admit": "admission_type_observation admit",
    "Surgical Same Day": "admission_type_surgical same day admission",
    "Urgent": "admission_type_urgent",
    "Unknown": "admission_type_nan",
}
insurance_options = {
    "Government": "insurance_government",
    "Medicaid": "insurance_medicaid",
    "Medicare": "insurance_medicare",
    "Other": "insurance_other",
    "Private": "insurance_private",
    "Unknown": "insurance_nan",
}
marital_status_options = {
    "Divorced": "marital_status_divorced",
    "Married": "marital_status_married",
    "Separated": "marital_status_separated",
    "Single": "marital_status_single",
    "Unknown": "marital_status_unknown (default)",
    "Widowed": "marital_status_widowed",
    "NaN": "marital_status_nan",  # Assuming this maps to marital_status_nan
}
diagnosis_features = [
    "diagnosis_circulatory",
    "diagnosis_digestive",
    "diagnosis_endocrine",
    "diagnosis_e_external_causes",
    "diagnosis_genitourinary",
    "diagnosis_injury",
    "diagnosis_mental",
    "diagnosis_musculoskeletal",
    "diagnosis_neoplasms",
    "diagnosis_nervous",
    "diagnosis_other",
    "diagnosis_respiratory",
    "diagnosis_v_supplementary",
]

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Admission")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=65, step=1)
    selected_gender = st.radio(
        "Gender", options=list(gender_options.keys()), index=0
    )  # Default to Female
    selected_admission_type = st.selectbox(
        "Admission Type", options=list(admission_type_options.keys()), index=4
    )  # Default to Emergency

with col2:
    st.subheader("Insurance & Marital Status")
    selected_insurance = st.selectbox(
        "Insurance", options=list(insurance_options.keys()), index=2
    )  # Default to Medicare
    selected_marital_status = st.selectbox(
        "Marital Status", options=list(marital_status_options.keys()), index=1
    )  # Default to Married

st.subheader("Diagnosis Categories")
# Create user-friendly labels for multiselect
diagnosis_labels = {
    feature: feature.replace("diagnosis_", "").replace("_", " ").title()
    for feature in diagnosis_features
}
# Special cases for labels
diagnosis_labels["diagnosis_e_external_causes"] = "External Causes (E-codes)"
diagnosis_labels["diagnosis_v_supplementary"] = "Supplementary (V-codes)"

# Use multiselect for diagnosis categories
selected_diagnoses_labels = st.multiselect(
    "Select all applicable primary diagnosis categories:",
    options=list(diagnosis_labels.values()),
    default=["Circulatory"],  # Default to Circulatory
)

# Map selected labels back to feature names
selected_diagnosis_features = [
    feature
    for feature, label in diagnosis_labels.items()
    if label in selected_diagnoses_labels
]

# --- Prediction ---
st.header("Prediction")

if st.button("Predict Readmission Risk"):
    # Construct the payload dictionary matching the API's Pydantic model keys
    payload = {}

    # 1. Age
    payload["age"] = float(age) if age is not None else None

    # 2. Gender (One-hot encode based on radio selection)
    selected_gender_feature = gender_options[selected_gender]
    for feature_name in gender_options.values():
        payload[feature_name] = 1 if feature_name == selected_gender_feature else 0

    # 3. Admission Type (One-hot encode based on selectbox selection)
    selected_admission_feature = admission_type_options[selected_admission_type]
    for feature_name in admission_type_options.values():
        payload[feature_name] = 1 if feature_name == selected_admission_feature else 0

    # 4. Insurance (One-hot encode based on selectbox selection)
    selected_insurance_feature = insurance_options[selected_insurance]
    for feature_name in insurance_options.values():
        payload[feature_name] = 1 if feature_name == selected_insurance_feature else 0

    # 5. Marital Status (One-hot encode based on selectbox selection)
    selected_marital_feature = marital_status_options[selected_marital_status]
    for feature_name in marital_status_options.values():
        payload[feature_name] = 1 if feature_name == selected_marital_feature else 0

    # 6. Diagnosis Features (from multiselect)
    for feature in diagnosis_features:
        payload[feature] = 1 if feature in selected_diagnosis_features else 0

    # 7. Feature '0' (Set to default 0 as it's not in the UI)
    # 7. Feature '0' (Include if expected by model, default to 0)
    # Assuming '0' is an expected feature based on previous API structure.
    # The updated API will ignore it if it's not in the loaded feature list.
    payload["0"] = 0

    # The API now handles feature validation, imputation, and aliases.
    # We just need to send the constructed payload dictionary.
    final_payload = payload

    st.write("Sending data to API:", final_payload)  # For debugging

    try:
        response = requests.post(API_URL, json=final_payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()
        probability = result.get("predicted_readmission_probability")

        if probability is not None:
            st.success(
                f"Predicted 30-Day Readmission Probability: **{probability:.4f}**"
            )
            # Add interpretation based on threshold if desired
            # threshold = 0.5 # Example threshold
            # if probability >= threshold:
            #     st.warning("High risk of readmission.")
            # else:
            #     st.info("Low risk of readmission.")
        else:
            st.error("Prediction failed. API response did not contain probability.")
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Connection Error: Could not connect to the API at {API_URL}. Is the Docker container running?"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        try:
            st.json(response.json())  # Show error details from API if available
        except:
            st.write("Could not parse error response from API.")

# --- Add instructions on how to run ---
st.sidebar.header("How to Run")
st.sidebar.code(
    f"""
# 1. Ensure the API Docker container is running:
#    docker start mimic-api-container
#
# 2. Install dashboard requirements:
#    pip install -r dashboard/requirements.txt
#
# 3. Run Streamlit app:
streamlit run dashboard/app.py
"""
)
st.sidebar.header("Model Features")
st.sidebar.info(
    "This dashboard uses the features identified in the loaded `readmission_model.pkl` file."
)
