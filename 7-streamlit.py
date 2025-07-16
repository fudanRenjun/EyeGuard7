import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from joblib import load
except Exception as e:
    st.error(f"⚠️ 导入 joblib.load 失败：{type(e).__name__}: {e}")
    raise

# --- Configuration ---
BINARY_MODEL_PATH = "2-12-stacking.pkl"
MULTI_MODEL_PATH = "7-12-stacking.pkl"

# 1) Model features
binary_features = [
    "TBIL", "TBA", "DBIL", "GLU", "LDH", "TG", "BUN",
    "ALP", "DD", "ALB", "RDW_SD", "TC"
]
multi_features = [
    "TBIL", "TBA", "UA", "GLU", "TC", "FIB", "P_LCR",
    "BUN", "DD", "TG", "RBC", "LDH", "PDW", "P_M"
]
all_features = sorted(set(binary_features + multi_features))

# 2) Feature name mapping
feature_labels = {
    "TBIL": "Total Bilirubin (\u03bcmol/L)",
    "TBA": "Total Bile Acid (\u03bcmol/L)",
    "DBIL": "Direct Bilirubin (\u03bcmol/L)",
    "GLU": "Glucose (mmol/L)",
    "LDH": "Lactate Dehydrogenase (U/L)",
    "TG": "Triglycerides (mmol/L)",
    "BUN": "Blood Urea Nitrogen (mmol/L)",
    "ALP": "Alkaline Phosphatase (U/L)",
    "DD": "D\u2011dimer (mg/L FEU)",
    "ALB": "Albumin (g/L)",
    "RDW_SD": "Red Cell Distribution Width\u2011SD (fL)",
    "TC": "Total Cholesterol (mmol/L)",
    "UA": "Uric Acid (\mmol/L)",
    "FIB": "Fibrinogen (g/L)",
    "P_LCR": "Platelet Large Cell Ratio (%)",
    "RBC": "Red Blood Cell Count (10^12/L)",
    "PDW": "Platelet Distribution Width (fL)",
    "P_M": "Monocyte Percentage (%)"
}

# 3) Disease class mapping
class_mapping = {
    0: "AMD",
    1: "ARC",
    2: "DR",
    3: "Glaucoma",
    4: "RD",
    5: "RP",
    6: "RVO"
}

# --- Streamlit UI ---
st.set_page_config(page_title="EyeGuard 7", layout="wide")
st.title("EyeGuard 7: Eye Disease Screening")

# 页面切换按钮（替代下拉框）
if "page" not in st.session_state:
    st.session_state.page = "Introduction"
    
with st.sidebar:
    st.write("## Page Navigation")
    if st.button("📖 Introduction"):
        st.session_state.page = "Introduction"
    if st.button("🩺 Start Screening"):
        st.session_state.page = "Start Screening"

page = st.session_state.page

# --- 页面内容 ---
if page == "Introduction":
    st.header("Project Overview")
    st.write("""
    **EyeGuard 7** is a web-based screening tool for early screening of eye diseases.

    **Screening purpose**:
    Our research team has constructed a novel disease screening model using Clinlabomics and machine learning to detect common blinding eye diseases.

    **Screening steps**:
    1. First, perform binary classification (Non-7 eye diseases vs. 7 eye diseases).
    2. If "7 eye diseases" is detected, perform 7 eye diseases classification to determine the subtype.

    **Instructions**:
    The model can be utilized for differential screening of the following diseases: Age-related Macular Degeneration, Age-related Cataract, Diabetic Retinopathy, Glaucoma, Retinal Detachment, Retinitis Pigmentosa, and Retinal Vein Occlusion.
    - Go to **Screening** to enter lab values.
    - Only when binary result is "Disease" will the subtype prediction run.
    """)
    
    st.info("Please click '🩺 Start Screening' on the left to begin using the model.")

elif page == "Start Screening":
    st.header("Enter Clinlabomics Indicators")

    try:
        binary_model = load(BINARY_MODEL_PATH)
        multi_model = load(MULTI_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    inputs = {}
    half = len(all_features) // 2

    for i, feat in enumerate(all_features):
        label = feature_labels.get(feat, feat)

        default_value = {
            "TBIL": 2.4,
            "TBA": 7,
            "DBIL": 4,
            "GLU": 10,
            "LDH": 170,
            "TG": 1.45,
            "BUN": 3,
            "ALP": 76,
            "DD": 0.22,
            "ALB": 48,
            "RDW_SD": 39.9,
            "TC": 6.13,
            "UA": 0.14,
            "FIB": 3.14,
            "P_LCR": 35.4,
            "RBC": 4.64,
            "PDW": 14.1,
            "P_M": 3.2
        }.get(feat, 0.0)

        if i < half:
            inputs[feat] = col1.number_input(label, value=default_value)
        else:
            inputs[feat] = col2.number_input(label, value=default_value)

    if st.button("Start Prediction"):
        df = pd.DataFrame([inputs])

        # --- Binary Classification ---
        Xb = df[binary_features]
        pred_bin = binary_model.predict(Xb)[0]
        prob_bin = binary_model.predict_proba(Xb)[0]

        st.subheader("Eye Diseases Screening Probabilities")

        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                bar_width = 0.3
                font_size_label = 11
                font_size_tick = 10
                font_size_anno = 10
                spine_width = 1.2

                fig1, ax1 = plt.subplots(figsize=(4.5, 5))
                x_positions = [0.5, 1.0]
                labels = ["Non-7 eye diseases", "7 eye diseases"]

                bars1 = ax1.bar(x_positions, prob_bin,
                                width=bar_width,
                                color=["#8fd9b6", "#ff9999"],
                                edgecolor='black',
                                linewidth=0.8)

                ax1.set_ylim(0, 1.1)
                ax1.set_ylabel("Probability", fontsize=font_size_label)
                ax1.set_title("Non-7 eye diseases vs. 7 eye diseases", fontsize=12)
                ax1.set_xticks(x_positions)
                ax1.set_xticklabels(labels, fontsize=font_size_tick)
                ax1.tick_params(axis='y', labelsize=font_size_tick)
                ax1.set_xlim(0.2, 1.3)

                for spine in ax1.spines.values():
                    spine.set_linewidth(spine_width)

                for bar in bars1:
                    height = bar.get_height()
                    ax1.annotate(f"{height:.1%}",
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 10),
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 fontsize=font_size_anno)

                plt.tight_layout()
                st.pyplot(fig1)

        if pred_bin == 0:
            st.success("Prediction: Non-7 eye diseases — No further screening needed.")
        else:
            st.warning("Prediction: 7 eye diseases detected — proceeding to subtype classification...")

            # --- Multi-class Classification ---
            Xm = df[multi_features]
            prob_multi = multi_model.predict_proba(Xm)[0]
            names = [class_mapping[c] for c in multi_model.classes_]

            st.subheader("Eye Diseases Subtype Screening Probabilities")

            with st.container():
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                bars2 = ax2.bar(names, prob_multi,
                                width=bar_width,
                                color="#6fa8dc",
                                edgecolor='black',
                                linewidth=0.8)

                ax2.set_ylim(0, 1.1)
                ax2.set_ylabel("Probability", fontsize=font_size_label)
                ax2.set_title("7 Eye Diseases Subtype Probabilities", fontsize=12)
                ax2.tick_params(axis='x', labelsize=font_size_tick, rotation=45)
                ax2.tick_params(axis='y', labelsize=font_size_tick)

                for spine in ax2.spines.values():
                    spine.set_linewidth(spine_width)

                for bar in bars2:
                    height = bar.get_height()
                    ax2.annotate(f"{height:.1%}",
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 10),
                                 textcoords="offset points",
                                 ha='center', va='bottom',
                                 fontsize=font_size_anno)

                plt.tight_layout()
                st.pyplot(fig2)

                st.caption("""
                **Abbreviations**:  
                - **AMD**: Age-related Macular Degeneration  
                - **ARC**: Age-related Cataract  
                - **DR**: Diabetic Retinopathy  
                - **RD**: Retinal Detachment  
                - **RP**: Retinitis Pigmentosa  
                - **RVO**: Retinal Vein Occlusion
                """)

