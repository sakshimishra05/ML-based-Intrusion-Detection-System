import os

import pandas as pd
import streamlit as st

from config import SAMPLE_DATA_DIR, UNKNOWN_THRESHOLD
from src.inference import InferenceError, run_detection
from src.preprocessor import PreprocessingError

NORMAL_SAMPLE = os.path.join(SAMPLE_DATA_DIR, "normal_traffic.csv")
ATTACK_SAMPLE = os.path.join(SAMPLE_DATA_DIR, "attack_traffic.csv")

st.set_page_config(
    page_title="ML-Based Intrusion Detection System",
    layout="wide",
)

st.title("ML-Based Intrusion Detection System")
st.caption(
    "CNN-LSTM based detection of normal traffic, known attacks, "
    "and potential unseen attacks"
)

st.subheader("Traffic Source")

uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    use_normal = st.button("Normal Traffic")
with col2:
    use_attack = st.button("Attack Traffic")

analyze = st.button("Analyze Traffic")


def load_sample_csv(path):
    return pd.read_csv(path)


def run_analysis(df_or_path, source_label):
    try:
        results, X, df_selected = run_detection(df_or_path)
    except PreprocessingError as exc:
        st.error(str(exc))
        return
    except InferenceError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return

    st.success(f"Analysis complete: {source_label}")

    total = len(results)
    normal_count = int((results["final_prediction"] == "Normal").sum())
    known_attack_count = int(
        (results["final_prediction"] != "Normal")
        & (results["final_prediction"] != "Unknown / Unseen Attack")
    ).sum()
    unknown_count = int(
        (results["final_prediction"] == "Unknown / Unseen Attack").sum()
    )

    normal_pct = normal_count / total * 100 if total else 0
    attack_pct = known_attack_count / total * 100 if total else 0
    unknown_pct = unknown_count / total * 100 if total else 0

    st.subheader("Traffic Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Traffic Records", total)
    m2.metric("Normal Traffic", normal_count)
    m3.metric("Known Attacks", known_attack_count)
    m4.metric("Unknown / Unseen Attacks", unknown_count)

    st.subheader("Distribution")
    d1, d2, d3 = st.columns(3)
    d1.metric("Normal %", f"{normal_pct:.1f}%")
    d2.metric("Attack %", f"{attack_pct:.1f}%")
    d3.metric("Unknown %", f"{unknown_pct:.1f}%")

    st.subheader("Detection Results")

    display = results.copy()
    display["binary_confidence"] = display["binary_confidence"].map(
        lambda v: f"{v:.1%}" if pd.notna(v) else "-"
    )
    display["final_confidence"] = display["final_confidence"].map(
        lambda v: f"{v:.1%}" if pd.notna(v) else "-"
    )
    display["multiclass_confidence"] = display["multiclass_confidence"].map(
        lambda v: f"{v:.1%}" if pd.notna(v) else "-"
    )
    display["multiclass_prediction"] = display["multiclass_prediction"].fillna("-")

    def highlight_unknown(row):
        if row["final_prediction"] == "Unknown / Unseen Attack":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(highlight_unknown, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detection Distribution")
    dist = results["final_prediction"].value_counts()
    st.bar_chart(dist)

    st.subheader("Download Results")
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction Results CSV",
        data=csv,
        file_name="ids_detection_results.csv",
        mime="text/csv",
    )

    st.info(
        "Unknown / Unseen Attack is an indicator based on low confidence "
        f"(< {UNKNOWN_THRESHOLD:.0%}) from the known-attack multiclass "
        "classifier. It is not a mathematically proven zero-day detection."
    )


if analyze:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV file: {exc}")
        else:
            run_analysis(df, "uploaded CSV")
    elif use_normal:
        run_analysis(NORMAL_SAMPLE, "Normal Traffic sample")
    elif use_attack:
        run_analysis(ATTACK_SAMPLE, "Attack Traffic sample")
    else:
        st.warning("Please upload a CSV or select a sample traffic file.")
