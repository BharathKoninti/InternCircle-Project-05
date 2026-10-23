import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).with_name("model.joblib")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

def train_and_save_model():
    """Create the same model pipeline if model.joblib has not been uploaded."""
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = sns.load_dataset("titanic")[["survived", *FEATURES]].dropna().copy()
    df["sex"] = (df["sex"] == "female").astype(int)
    X, y = df[FEATURES], df["survived"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model
st.title("Titanic Survival Predictor")
st.write("Explore how a simple machine-learning model estimates survival probability from passenger details.")

@st.cache_resource
def load_model():
    """Load the trained classifier once and reuse it."""
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)

st.subheader("Passenger details")
pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2)
sex_label = st.radio("Sex", options=["female", "male"], horizontal=True)
age = st.slider("Age (years)", min_value=1, max_value=80, value=30)
fare = st.slider("Fare paid (dataset units)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)

if st.button("Predict survival probability", type="primary"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file is missing. Run the training notebook to create model.joblib, then place it beside app.py.")
        st.stop()
    except Exception as exc:
        st.error(f"The model could not be loaded: {exc}")
        st.stop()

    try:
        input_row = pd.DataFrame(
            [[pclass, 1 if sex_label == "female" else 0, age, fare]],
            columns=FEATURES
        )
        probability = float(model.predict_proba(input_row)[0][1])
        prediction = int(model.predict(input_row)[0])
        st.metric("Estimated chance of survival", f"{probability:.0%}")
        if prediction == 1:
            st.success("The model predicts: survived.")
        else:
            st.info("The model predicts: did not survive.")
    except Exception as exc:
        st.error(f"Could not make a prediction from these inputs: {exc}")

st.caption(
    "Educational example only. This model is trained on a small historical Titanic dataset; "
    "it is not advice and should not be used for real-world decisions."
)
