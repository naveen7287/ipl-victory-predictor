import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ------------------ Load model ------------------
def load_model():
    # common names first
    candidates = ["model.pkl", "pipeline.pkl", "model.sav", "model.joblib"]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "rb") as fh:
                    return pickle.load(fh), c
            except Exception:
                pass
    # recursive search
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith((".pkl", ".joblib", ".sav")):
                path = os.path.join(root, f)
                try:
                    with open(path, "rb") as fh:
                        return pickle.load(fh), path
                except Exception:
                    pass
    return None, None

model, model_path = load_model()
# ---- BEGIN: patch ColumnTransformer passthrough string -> real transformer ----
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def replace_string_passthroughs(est):
    """
    Fixes broken transformers (like string 'passthrough') inside ColumnTransformer
    so sklearn 1.7.2 can run the pipeline created under sklearn 1.3.0.
    """
    # If Pipeline → go into its steps
    if isinstance(est, Pipeline):
        for name, step in est.steps:
            replace_string_passthroughs(step)
        return

    # If ColumnTransformer → inspect transformers
    if isinstance(est, ColumnTransformer):
        trs_attr = "transformers_" if hasattr(est, "transformers_") else "transformers"
        try:
            trs = getattr(est, trs_attr)
        except Exception:
            trs = getattr(est, "transformers", [])

        new_trs = []
        for name, transformer, cols in trs:
            # If transformer is a STRING → bad, replace with real passthrough transformer
            if isinstance(transformer, str):
                if transformer.lower() == "passthrough":
                    new_transformer = FunctionTransformer(lambda x: x)
                else:
                    new_transformer = FunctionTransformer(lambda x: x)
                new_trs.append((name, new_transformer, cols))
            else:
                replace_string_passthroughs(transformer)
                new_trs.append((name, transformer, cols))

        # Set back corrected transformers
        try:
            if hasattr(est, "transformers_"):
                est.transformers_ = new_trs
            else:
                est.transformers = new_trs
        except Exception:
            pass

# apply fix to the loaded model
try:
    replace_string_passthroughs(model)
except Exception as e:
    print("Warning: passthrough patch failed:", e)
# ---- END PATCH ----

if model is None:
    st.error("Model not found or could not be loaded. Make sure a valid .pkl/.joblib file is present.")
    st.stop()

st.sidebar.markdown(f"**Loaded model:** `{model_path}`")


# ------------------ Helpers ------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def find_column_transformer(estimator):
    if estimator is None:
        return None
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            ct = find_column_transformer(step)
            if ct is not None:
                return ct
    if isinstance(estimator, ColumnTransformer):
        return estimator
    for attr in ("transformers_", "named_transformers_", "transformers"):
        if hasattr(estimator, attr):
            try:
                trs = getattr(estimator, attr)
                if isinstance(trs, list):
                    for item in trs:
                        if len(item) >= 3:
                            transformer = item[1]
                            ct = find_column_transformer(transformer)
                            if ct is not None:
                                return ct
            except Exception:
                pass
    return None

def get_categories_from_ct(ct):
    cats = {}
    if ct is None:
        return cats
    try:
        transformers = getattr(ct, "transformers_", None) or getattr(ct, "transformers", [])
        for name, transformer, cols in transformers:
            enc = None
            if hasattr(transformer, "named_steps"):
                for step in transformer.named_steps.values():
                    if isinstance(step, OneHotEncoder) or hasattr(step, "categories_"):
                        enc = step
                        break
            elif isinstance(transformer, OneHotEncoder) or hasattr(transformer, "categories_"):
                enc = transformer
            if enc is not None and hasattr(enc, "categories_"):
                try:
                    for col, cat in zip(cols, enc.categories_):
                        cats[col] = np.array(cat, dtype=object)
                except Exception:
                    cats[f"_enc_{name}"] = np.hstack(enc.categories_)
    except Exception:
        pass
    return cats

def safe_align_categories(model, input_df):
    ct = find_column_transformer(model)
    if ct is None:
        return input_df
    cats = get_categories_from_ct(ct)
    if not cats:
        return input_df
    df = input_df.copy()
    for col, known_vals in list(cats.items()):
        if col in df.columns:
            fallback = known_vals[0] if len(known_vals) > 0 else None
            df[col] = df[col].apply(lambda v: v if v in known_vals else fallback)
    return df

def safe_predict_proba(model, input_df):
    # 1) try passing DataFrame
    try:
        return model.predict_proba(input_df)
    except AttributeError as e:
        print("safe_predict_proba DataFrame AttributeError:", e)
    except Exception as e:
        print("safe_predict_proba DataFrame raised:", type(e).__name__, str(e))

    # 2) try numpy array (use feature_names_in_ if available)
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            if all(c in input_df.columns for c in cols):
                X = input_df[cols].to_numpy()
            else:
                X = input_df.to_numpy()
        else:
            X = input_df.to_numpy()
        try:
            return model.predict_proba(X)
        except Exception as e2:
            print("safe_predict_proba numpy predict_proba failed:", type(e2).__name__, str(e2))
    except Exception as e:
        print("safe_predict_proba preparing numpy fallback failed:", type(e).__name__, str(e))

    # 3) fallback: predict() and convert to probabilities (binary)
    try:
        pred = model.predict(input_df if hasattr(model, "predict") else X)
        probs = []
        for p in pred:
            if p in (0, "0", False):
                probs.append([1.0, 0.0])
            else:
                probs.append([0.0, 1.0])
        return np.array(probs)
    except Exception as e:
        print("safe_predict_proba final fallback failed:", type(e).__name__, str(e))
        raise


# ------------------ Streamlit UI ------------------
st.title("IPL Victory Probability Estimator")

teams = [
    "Chennai Super Kings", "Royal Challengers Bangalore", "Sunrisers Hyderabad",
    "Mumbai Indians", "Kolkata Knight Riders", "Gujarat Titans",
    "Rajasthan Royals", "Delhi Capitals", "Punjab Kings", "Lucknow Super Giants"
]

cities = [
    "Mumbai", "Kolkata", "Chennai", "Delhi", "Hyderabad",
    "Pune", "Bangalore", "Ahmedabad", "Jaipur", "Abu Dhabi", "Dubai"
]

col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("Select the batting team", teams, index=0)
    target = st.number_input("Target Score", min_value=0, step=1, value=150)
with col2:
    bowling_team = st.selectbox("Select the bowling team", teams, index=1)
    score = st.number_input("Score", min_value=0, step=1, value=120)
with col3:
    city = st.selectbox("Select the host city", cities, index=0)
    overs = st.number_input("Overs Done", min_value=0.0, step=0.1, value=15.0)
    wickets = st.number_input("Wickets Fell", min_value=0, max_value=10, step=1, value=3)

if st.button("Predict Probabilities"):
    # compute features
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - int(wickets)
    crr = float(score) / float(overs) if overs > 0 else 0.0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

    # build DataFrame with commonly expected names (exact spelling/casing matters)
    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [city],

        "target_left": [runs_left],
        "Remaining Balls": [balls_left],
        "Score": [score],
        "Wickets": [wickets],

        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets_left": [wickets_left],
        "crr": [crr],
        "rrr": [rrr]
    })

    # align categories (avoid unseen category errors)
    aligned = safe_align_categories(model, input_df)

    try:
        result = safe_predict_proba(model, aligned)
    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.stop()

    try:
        loss = float(result[0][0])
        win = float(result[0][1])
    except Exception:
        if len(result.shape) == 1:
            prob_win = float(result[0])
            win = prob_win
            loss = 1.0 - prob_win
        else:
            st.error("Unexpected model output shape: " + str(result.shape))
            st.write("Raw output:", result)
            st.stop()

    st.metric(label=f"{batting_team} Win Probability", value=f"{round(win*100,2)}%")
    st.metric(label=f"{bowling_team} Win Probability", value=f"{round(loss*100,2)}%")

    if st.checkbox("Show debug info"):
        st.write("Input dataframe (aligned):")
        st.write(aligned)
        try:
            ct = find_column_transformer(model)
            st.write("Found ColumnTransformer:", bool(ct))
            if ct is not None:
                st.write("Categories found for encoders (sample):", list(get_categories_from_ct(ct).items())[:5])
        except Exception as e:
            st.write("Inspector error:", e)
