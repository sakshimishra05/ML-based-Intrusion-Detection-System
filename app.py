import pandas as pd
import streamlit as st

from config import BATCH_ATTACK_THRESHOLD, UNKNOWN_THRESHOLD
from src.inference import InferenceError, run_detection
from src.preprocessor import PreprocessingError


def get_batch_detection_result(binary_predictions):
    total_records = len(binary_predictions)
    attack_percentage = (
        (binary_predictions == "ATTACK").sum() / total_records * 100
        if total_records
        else 0
    )
    attack_detected = attack_percentage >= BATCH_ATTACK_THRESHOLD * 100

    if attack_detected:
        supporting_text = (
            f"{attack_percentage:.1f}% of traffic records were classified "
            "as attack traffic."
        )
    else:
        supporting_text = (
            f"{attack_percentage:.1f}% of traffic records were classified "
            "as attack traffic, below the "
            f"{BATCH_ATTACK_THRESHOLD:.0%} alert threshold."
        )

    return attack_detected, supporting_text


st.set_page_config(
    page_title="CyberShield | Network Intrusion Detection System",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ids-bg: #070d18;
        --ids-panel: rgba(13, 23, 39, .88);
        --ids-panel-raised: #101d30;
        --ids-line: rgba(115, 151, 190, .19);
        --ids-ink: #e7f0fb;
        --ids-muted: #93a6bc;
        --ids-cyan: #36d8f2;
        --ids-blue: #3978ff;
        --ids-green: #36d399;
        --ids-red: #ff5b66;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 18% -10%, rgba(24, 93, 143, .24), transparent 38rem),
            radial-gradient(ellipse at 92% 18%, rgba(16, 106, 121, .10), transparent 30rem),
            var(--ids-bg);
        color: var(--ids-ink);
    }

    [data-testid="stAppViewContainer"]::before {
        position: fixed;
        z-index: 0;
        inset: 0;
        background-image:
            linear-gradient(rgba(99, 146, 190, .035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99, 146, 190, .035) 1px, transparent 1px);
        background-size: 44px 44px;
        content: "";
        pointer-events: none;
        mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    [data-testid="stHeader"] {
        z-index: 2;
        background: rgba(7, 13, 24, .72);
        backdrop-filter: blur(14px);
    }

    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
        max-width: 1280px;
        padding: 1.55rem 2.25rem 4rem;
    }

    h1, h2, h3, label {
        color: var(--ids-ink);
        letter-spacing: -0.035em;
    }

    h2 {
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.25rem;
        font-weight: 680;
    }

    p, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
        color: var(--ids-muted);
    }

    .ids-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.15rem;
        margin: .2rem 0 1.35rem;
        border: 1px solid var(--ids-line);
        border-radius: 15px;
        background: linear-gradient(105deg, rgba(15, 28, 47, .94), rgba(11, 22, 38, .82));
        box-shadow: 0 10px 35px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .025);
        backdrop-filter: blur(16px);
    }

    .ids-brand-icon {
        display: grid;
        flex: 0 0 48px;
        width: 48px;
        height: 48px;
        place-items: center;
        border: 1px solid rgba(54, 216, 242, .3);
        border-radius: 13px;
        color: var(--ids-cyan);
        background: linear-gradient(145deg, rgba(54, 216, 242, .15), rgba(57, 120, 255, .1));
        box-shadow: 0 0 24px rgba(54, 216, 242, .08);
    }

    .ids-brand-icon svg {
        width: 27px;
        height: 27px;
    }

    .ids-brand-title {
        color: var(--ids-ink);
        font-size: clamp(1rem, 2vw, 1.35rem);
        font-weight: 720;
        letter-spacing: -.025em;
        line-height: 1.3;
    }

    .ids-brand-subtitle {
        margin-top: .25rem;
        color: var(--ids-muted);
        font-size: .82rem;
        letter-spacing: .01em;
    }

    .ids-header-pill {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        flex: 0 0 auto;
        padding: .48rem .7rem;
        margin-left: auto;
        border: 1px solid rgba(54, 216, 242, .2);
        border-radius: 999px;
        color: #a8eefa;
        background: rgba(31, 127, 151, .1);
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .09em;
        white-space: nowrap;
    }

    .ids-header-pill span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--ids-cyan);
        box-shadow: 0 0 10px rgba(54, 216, 242, .8);
    }

    .ids-page-intro {
        margin: 0 0 1.4rem;
        color: var(--ids-muted);
        font-size: .92rem;
    }

    [class*="st-key-ids-upload-panel"] {
        padding: 1.2rem 1.35rem .65rem;
        border: 1px solid var(--ids-line);
        border-radius: 17px;
        background: linear-gradient(145deg, rgba(16, 29, 48, .9), rgba(10, 20, 35, .9));
        box-shadow: 0 16px 38px rgba(0, 0, 0, .17), inset 0 1px rgba(255, 255, 255, .025);
    }

    .ids-upload-heading {
        display: flex;
        align-items: center;
        gap: .8rem;
        margin: 0 0 .9rem;
    }

    .ids-upload-icon {
        display: grid;
        flex: 0 0 40px;
        width: 40px;
        height: 40px;
        place-items: center;
        border: 1px solid rgba(54, 216, 242, .2);
        border-radius: 11px;
        color: var(--ids-cyan);
        background: rgba(54, 216, 242, .08);
    }

    .ids-upload-icon svg {
        width: 21px;
        height: 21px;
    }

    .ids-upload-title {
        color: var(--ids-ink);
        font-size: 1rem;
        font-weight: 680;
    }

    .ids-upload-copy {
        margin-top: .2rem;
        color: var(--ids-muted);
        font-size: .82rem;
    }

    [data-testid="stFileUploader"] {
        padding: .7rem;
        border: 1px solid rgba(92, 136, 178, .18);
        border-radius: 13px;
        background: rgba(6, 14, 26, .62);
    }

    [data-testid="stFileUploader"] label {
        color: var(--ids-muted);
        font-size: .82rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 118px;
        border: 1px dashed rgba(54, 173, 235, .54);
        border-radius: 11px;
        background: linear-gradient(135deg, rgba(24, 72, 105, .17), rgba(11, 30, 50, .4));
        transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--ids-cyan);
        background: rgba(21, 76, 105, .24);
        box-shadow: 0 0 20px rgba(54, 216, 242, .08);
    }

    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {
        color: #c4d6e8 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #91a8c0 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid rgba(54, 216, 242, .42);
        border-radius: 8px;
        color: #dffaff;
        background: rgba(25, 112, 145, .2);
    }

    [data-testid="stFileChip"] {
        max-width: min(100%, 28rem);
        padding: .55rem .7rem;
        border: 1px solid rgba(54, 211, 153, .25);
        border-radius: 10px;
        color: #cdeee1;
        background: rgba(32, 132, 100, .12);
        box-shadow: 0 4px 14px rgba(0, 0, 0, .12);
    }

    [data-testid="stFileChipName"] {
        min-width: 0;
        color: #d6e8f4;
    }

    [data-testid="stFileChip"] p,
    [data-testid="stFileChip"] small {
        color: #91a8c0 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileChipDeleteBtn"] {
        border: 1px solid rgba(54, 216, 242, .24);
        border-radius: 7px;
        color: #c1eaf2;
        background: rgba(54, 216, 242, .08);
    }

    [data-testid="stButton"] > button,
    [data-testid="stBaseButton-primary"] {
        min-height: 3.45rem;
        border: 1px solid #45d9ff !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        background: linear-gradient(105deg, #04a9d8 0%, #2778f6 55%, #3b61ed 100%) !important;
        box-shadow: 0 0 0 1px rgba(55, 193, 255, .12), 0 8px 26px rgba(17, 127, 234, .27) !important;
        font-size: .98rem !important;
        font-weight: 750 !important;
        letter-spacing: .01em;
        opacity: 1 !important;
        transition: transform .16s ease, box-shadow .16s ease, filter .16s ease !important;
    }

    [data-testid="stButton"] > button p,
    [data-testid="stBaseButton-primary"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stButton"] > button:hover:not(:disabled),
    [data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        border-color: #93f2ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 24px rgba(54, 216, 242, .3), 0 12px 30px rgba(17, 127, 234, .32) !important;
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    [data-testid="stButton"] > button:focus-visible,
    [data-testid="stBaseButton-primary"]:focus-visible {
        outline: 3px solid rgba(54, 216, 242, .55) !important;
        outline-offset: 3px;
    }

    [data-testid="stButton"] > button:active:not(:disabled),
    [data-testid="stBaseButton-primary"]:active:not(:disabled) {
        filter: brightness(.92);
        transform: translateY(1px);
    }

    [data-testid="stButton"] > button:disabled,
    [data-testid="stBaseButton-primary"]:disabled {
        border-color: #3f536b !important;
        color: #b6c4d3 !important;
        background: #26364a !important;
        box-shadow: none !important;
        opacity: 1 !important;
        cursor: not-allowed;
    }

    [data-testid="stSpinner"] {
        color: var(--ids-cyan);
    }

    [data-testid="stSpinner"] p {
        color: #d8f6ff !important;
        font-weight: 650;
    }

    .ids-result {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.1rem 1.25rem;
        margin: 1.35rem 0 1.7rem;
        border: 1px solid;
        border-radius: 15px;
        background: rgba(15, 27, 44, .9);
        box-shadow: 0 12px 30px rgba(0, 0, 0, .18);
    }

    .ids-result-normal {
        border-color: rgba(54, 211, 153, .34);
        box-shadow: 0 0 24px rgba(54, 211, 153, .055), 0 12px 30px rgba(0, 0, 0, .18);
    }

    .ids-result-attack {
        border-color: rgba(255, 91, 102, .52);
        box-shadow: 0 0 28px rgba(255, 91, 102, .11), 0 12px 30px rgba(0, 0, 0, .18);
    }

    .ids-result-icon {
        display: grid;
        flex: 0 0 44px;
        width: 44px;
        height: 44px;
        place-items: center;
        border-radius: 13px;
    }

    .ids-result-icon svg {
        width: 24px;
        height: 24px;
    }

    .ids-result-normal .ids-result-icon {
        color: var(--ids-green);
        background: rgba(54, 211, 153, .12);
    }

    .ids-result-attack .ids-result-icon {
        color: var(--ids-red);
        background: rgba(255, 91, 102, .13);
        animation: threat-pulse 2.3s ease-in-out infinite;
    }

    .ids-result-kicker {
        margin-bottom: .22rem;
        color: var(--ids-muted);
        font-size: .68rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .ids-result-title {
        color: var(--ids-ink);
        font-size: 1.04rem;
        font-weight: 760;
        letter-spacing: -.01em;
    }

    .ids-result-copy {
        margin-top: .24rem;
        color: #a6b8ca;
        font-size: .83rem;
        line-height: 1.5;
    }

    .ids-result-percent {
        flex: 0 0 auto;
        margin-left: auto;
        color: var(--ids-cyan);
        font-size: clamp(1.4rem, 3vw, 2rem);
        font-variant-numeric: tabular-nums;
        font-weight: 780;
        letter-spacing: -.05em;
        white-space: nowrap;
    }

    .ids-result-attack .ids-result-percent {
        color: #ff7d75;
    }

    @keyframes threat-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 91, 102, 0); }
        50% { box-shadow: 0 0 0 7px rgba(255, 91, 102, .08); }
    }

    [class*="st-key-ids-summary-grid"] [data-testid="stHorizontalBlock"] {
        align-items: stretch;
        gap: .85rem;
    }

    .ids-stat-card {
        display: flex;
        min-height: 142px;
        flex-direction: column;
        padding: 1rem;
        border: 1px solid var(--ids-line);
        border-radius: 14px;
        background: linear-gradient(150deg, rgba(17, 31, 51, .94), rgba(10, 19, 33, .92));
        box-shadow: 0 8px 22px rgba(0, 0, 0, .14), inset 0 1px rgba(255, 255, 255, .025);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }

    .ids-stat-card:hover {
        border-color: rgba(54, 216, 242, .35);
        box-shadow: 0 10px 28px rgba(0, 0, 0, .2), 0 0 20px rgba(54, 216, 242, .055);
        transform: translateY(-2px);
    }

    .ids-stat-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: .5rem;
    }

    .ids-stat-label {
        color: #b8c7d9;
        font-size: .81rem;
        font-weight: 620;
        line-height: 1.35;
    }

    .ids-stat-icon {
        display: grid;
        flex: 0 0 32px;
        width: 32px;
        height: 32px;
        place-items: center;
        border: 1px solid rgba(54, 216, 242, .16);
        border-radius: 9px;
        color: var(--ids-cyan);
        background: rgba(54, 216, 242, .08);
    }

    .ids-stat-icon svg {
        width: 17px;
        height: 17px;
    }

    .ids-stat-icon-normal {
        border-color: rgba(54, 211, 153, .2);
        color: var(--ids-green);
        background: rgba(54, 211, 153, .09);
    }

    .ids-stat-icon-alert {
        border-color: rgba(255, 91, 102, .22);
        color: var(--ids-red);
        background: rgba(255, 91, 102, .09);
    }

    .ids-stat-icon-unknown {
        border-color: rgba(109, 154, 255, .2);
        color: #91b4ff;
        background: rgba(57, 120, 255, .09);
    }

    .ids-stat-value {
        margin-top: auto;
        padding-top: .7rem;
        color: #f1f7ff;
        font-size: 1.8rem;
        font-variant-numeric: tabular-nums;
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1;
    }

    .ids-stat-description {
        margin-top: .42rem;
        color: #71859d;
        font-size: .71rem;
        line-height: 1.3;
    }

    [data-testid="stMetric"] {
        height: 100%;
        padding: .85rem 1rem;
        border: 1px solid var(--ids-line);
        border-radius: 12px;
        background: rgba(14, 25, 42, .86);
    }

    [data-testid="stMetricLabel"] {
        color: #aebed0;
    }

    [data-testid="stMetricValue"] {
        color: #eaf5ff;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        background: rgba(15, 28, 45, .8);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--ids-line);
        border-radius: 13px;
        background: rgba(14, 25, 42, .86);
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: var(--ids-ink);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--ids-line);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stMarkdownContainer"] hr {
        border-color: var(--ids-line);
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-right: 1.25rem;
            padding-left: 1.25rem;
        }

        [class*="st-key-ids-summary-grid"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        [class*="st-key-ids-summary-grid"] [data-testid="column"] {
            min-width: calc(50% - .5rem);
            flex: 1 1 calc(50% - .5rem);
        }
    }

    @media (max-width: 580px) {
        [data-testid="stMainBlockContainer"] {
            padding: .8rem .8rem 2rem;
        }

        .ids-header {
            align-items: flex-start;
            flex-wrap: wrap;
            gap: .7rem;
            padding: .85rem;
        }

        .ids-brand-icon {
            flex-basis: 40px;
            width: 40px;
            height: 40px;
        }

        .ids-brand-copy {
            flex: 1 1 calc(100% - 52px);
        }

        .ids-header-pill {
            margin-left: 48px;
        }

        [class*="st-key-ids-upload-panel"] {
            padding: .9rem .75rem .45rem;
        }

        .ids-result {
            align-items: flex-start;
            gap: .7rem;
            padding: .9rem;
        }

        .ids-result-icon {
            flex-basis: 36px;
            width: 36px;
            height: 36px;
        }

        .ids-result-percent {
            align-self: center;
        }

        [class*="st-key-ids-summary-grid"] [data-testid="column"] {
            min-width: 100%;
            flex-basis: 100%;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
            animation-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="ids-header">
        <div class="ids-brand-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3 19 6v5c0 4.7-2.9 8.1-7 10-4.1-1.9-7-5.3-7-10V6l7-3Z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
        </div>
        <div class="ids-brand-copy">
            <div class="ids-brand-title">CyberShield | Network Intrusion Detection System</div>
            <div class="ids-brand-subtitle">AI-Powered Real-Time Threat Intelligence &amp; Network Analysis</div>
        </div>
        <div class="ids-header-pill"><span></span> BATCH CSV ANALYSIS</div>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="ids-page-intro">Inspect traffic records and surface potential threats with the trained detection models.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True, key="ids-upload-panel"):
    st.markdown(
        """
        <div class="ids-upload-heading">
            <div class="ids-upload-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 16V4m0 0L8 8m4-4 4 4"/>
                    <path d="M20 16.5v2A1.5 1.5 0 0 1 18.5 20h-13A1.5 1.5 0 0 1 4 18.5v-2"/>
                </svg>
            </div>
            <div>
                <div class="ids-upload-title">Upload Network Traffic Dataset</div>
                <div class="ids-upload-copy">Drag &amp; drop your CSV file here or browse files</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="Choose a CSV file with the required ACO-selected network features.",
        label_visibility="collapsed",
    )

analyze = st.button(
    "Analyze Traffic",
    type="primary",
    icon=":material/search:",
    use_container_width=True,
)


def run_analysis(df, source_label):
    try:
        results, X, df_selected = run_detection(df)
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
        (
            (results["final_prediction"] != "Normal")
            & (results["final_prediction"] != "Unknown / Unseen Attack")
        ).sum()
    )
    unknown_count = int(
        (results["final_prediction"] == "Unknown / Unseen Attack").sum()
    )

    normal_pct = normal_count / total * 100 if total else 0
    attack_pct = known_attack_count / total * 100 if total else 0
    unknown_pct = unknown_count / total * 100 if total else 0

    # ---- Overall detection result (binary only) ---
    attack_detected, supporting_text = get_batch_detection_result(
        results["binary_prediction"]
    )
    binary_attack_count = int(
        (results["binary_prediction"] == "ATTACK").sum()
    )
    binary_attack_percentage = (
        binary_attack_count / total * 100 if total else 0
    )
    result_class = "attack" if attack_detected else "normal"
    result_title = (
        "ATTACK DETECTED" if attack_detected else "NO ATTACK DETECTED"
    )
    result_icon = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><path d="M10.3 3.9 2.8 17a2 2 0 0 0 1.7 3h15a2 '
        '2 0 0 0 1.7-3l-7.5-13a2 2 0 0 0-3.4-.1Z"/><path d="M12 9v4m0 '
        '3h.01"/></svg>'
        if attack_detected
        else
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true"><path d="M12 3 19 6v5c0 4.7-2.9 8.1-7 10-4.1-1.9-7-5.3-7-10V6l7-3Z"/>'
        '<path d="m9 12 2 2 4-4"/></svg>'
    )
    st.markdown(
        f"""
        <section class="ids-result ids-result-{result_class}" role="status"
                 aria-label="Detection result: {result_title}">
            <div class="ids-result-icon">{result_icon}</div>
            <div>
                <div class="ids-result-kicker">Batch detection result</div>
                <div class="ids-result-title">Detection Result: {result_title}</div>
                <div class="ids-result-copy">{supporting_text}</div>
            </div>
            <div class="ids-result-percent">{binary_attack_percentage:.1f}%</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ---- Detection Summary (main result) ----
    st.subheader("Detection Summary")
    summary_cards = [
        (
            "Total Traffic Records",
            total,
            "Records analyzed",
            "cyan",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M4 6.5 12 3l8 3.5-8 3.5L4 6.5Z"/><path d="m4 12 8 3.5 8-3.5M4 17.5 12 21l8-3.5"/></svg>',
        ),
        (
            "Normal Traffic",
            normal_count,
            "Classified as normal",
            "normal",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 3 19 6v5c0 4.7-2.9 8.1-7 10-4.1-1.9-7-5.3-7-10V6l7-3Z"/>'
            '<path d="m9 12 2 2 4-4"/></svg>',
        ),
        (
            "Known Attacks",
            known_attack_count,
            "Known attack classifications",
            "alert",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M10.3 3.9 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3l-7.5-13a2 2 0 0 0-3.4-.1Z"/>'
            '<path d="M12 9v4m0 3h.01"/></svg>',
        ),
        (
            "Unknown / Zero-Day Attacks",
            unknown_count,
            "Potential unseen indicators",
            "unknown",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.5 2.5 0 1 1 4.4 1.6c-1.2 1.1-2 1.4-2 3"/>'
            '<path d="M12 17h.01"/></svg>',
        ),
    ]
    with st.container(key="ids-summary-grid"):
        columns = st.columns(4, gap="medium")
        for column, (label, value, description, tone, icon) in zip(
            columns, summary_cards
        ):
            with column:
                st.markdown(
                    f"""
                    <div class="ids-stat-card">
                        <div class="ids-stat-top">
                            <div class="ids-stat-label">{label}</div>
                            <div class="ids-stat-icon ids-stat-icon-{tone}" aria-hidden="true">{icon}</div>
                        </div>
                        <div class="ids-stat-value">{value:,}</div>
                        <div class="ids-stat-description">{description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.subheader("Traffic Distribution")
    d1, d2, d3 = st.columns(3)
    d1.metric("Normal", f"{normal_pct:.1f}%")
    d2.metric("Known Attack", f"{attack_pct:.1f}%")
    d3.metric("Unknown / Unseen", f"{unknown_pct:.1f}%")

    # Bar chart of detection distribution
    dist = results["final_prediction"].value_counts()
    st.bar_chart(dist)

    # ---- Attack type breakdown ----
    st.subheader("Attack Type Breakdown")
    attack_rows = results[results["final_prediction"] != "Normal"]
    if len(attack_rows) > 0:
        breakdown = (
            attack_rows["final_prediction"]
            .value_counts()
            .reset_index()
        )
        breakdown.columns = ["Attack Type", "Count"]
        breakdown["Percentage"] = (
            breakdown["Count"] / total * 100
        ).round(1).astype(str) + "%"
        st.table(breakdown)
    else:
        st.info("No attacks detected.")

    # ---- Average confidence ----
    st.subheader("Average Confidence")
    c1, c2 = st.columns(2)
    binary_avg = results["binary_confidence"].mean()
    c1.metric("Binary Detection Confidence", f"{binary_avg:.1%}")

    attack_only = results[results["binary_prediction"] == "ATTACK"]
    if len(attack_only) > 0:
        mc_avg = attack_only["multiclass_confidence"].mean()
        c2.metric("Multiclass Classification Confidence", f"{mc_avg:.1%}")
    else:
        c2.metric("Multiclass Classification Confidence", "N/A")

    # ---- Unknown/unseen disclaimer ----
    st.info(
        "Unknown / Unseen Attack is an indicator based on low confidence "
        f"(< {UNKNOWN_THRESHOLD:.0%}) from the known-attack multiclass "
        "classifier. It is not a mathematically proven zero-day detection."
    )

    # ---- Detailed row predictions (expandable) ----
    with st.expander("View Detailed Row Predictions", expanded=False):
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
                return [
                    "background-color: #342b1d; color: #f5d894"
                ] * len(row)
            return [""] * len(row)

        styled = display.style.apply(highlight_unknown, axis=1)
        st.dataframe(
            styled,
            width='stretch',
            hide_index=True,
        )

    # ---- Download ----
    st.subheader("Download Results")
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction Results CSV",
        data=csv,
        file_name="ids_detection_results.csv",
        mime="text/csv",
    )


if analyze:
    if uploaded_file is not None:
        with st.spinner("Analyzing Network Traffic..."):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Failed to read CSV file: {exc}")
            else:
                run_analysis(df, "uploaded CSV")
    else:
        st.warning("Please upload a CSV file to begin.")
