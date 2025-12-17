import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Arshyan & Moiz DM Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: bold; color: #1E3A8A; }
    .sub-header { font-size: 20px; color: #555; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<p class="main-header">📊 Arshyan & Moiz Data Mining Project</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Sector Stock Market Analysis & Prediction System</p>', unsafe_allow_html=True)
st.write("---")

# --- SIDEBAR: DATASET SELECTION ---
st.sidebar.header("📁 Project Controls")
dataset_choice = st.sidebar.radio("Select Stock Dataset:", ["FAUJI", "OGDCL"])

st.sidebar.info(f"**Current Status:**\nSelected: {dataset_choice}")

# --- SECTION 1: PREPROCESSING SUMMARY ---
st.header(f"1. Preprocessing Summary: {dataset_choice}")

# Dynamic CSV path
processed_csv = f"{dataset_choice}_Processed_Data.csv"

if os.path.exists(processed_csv):
    try:
        df = pd.read_csv(processed_csv)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"✅ Loaded {processed_csv}")
            st.markdown("""
            **Pipeline Applied:**
            * **Cleaning:** Null handling (ffill/bfill)
            * **Scaling:** Volatility & RSI standardized
            * **Engineering:** 50-Day MA, 7-Day Volatility
            * **Clustering:** K-Means Regime Labeling
            """)
        
        with col2:
            st.write("**Data Preview (First 5 Rows):**")
            # FIX: Removed invalid width arguments. Letting Streamlit handle default width.
            st.dataframe(df.head(5), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
else:
    st.warning(f"⚠️ File not found: `{processed_csv}`. Please ensure it exists in the directory.")

st.write("---")

# --- SECTION 2: MODEL SELECTION ---
st.header("2. Model Application & Results")

model_options = ["Logistic Regression", "Random Forest", "SVM", "Voting Ensemble"]
selected_model = st.selectbox("Choose Model Architecture:", model_options)

# Construct filenames dynamically based on selection
# Note: This matches the naming convention in your screenshots
model_key = selected_model.replace(' ', '_')
model_filename = f"{dataset_choice}_{model_key}_Model.pkl"
dashboard_filename = f"{dataset_choice}_{model_key}_Dashboard.html"

# --- SECTION 3: EXECUTION & DASHBOARD ---
if st.button(f"🚀 Deploy {selected_model}"):
    
    # 3.1: Verify Files Exist
    model_exists = os.path.exists(model_filename)
    dashboard_exists = os.path.exists(dashboard_filename)
    
    if not model_exists:
        st.error(f"❌ **Missing Model File:** `{model_filename}`")
        st.info("Please make sure you have trained this specific model and the .pkl file is in the folder.")
    
    elif not dashboard_exists:
        st.error(f"❌ **Missing Dashboard File:** `{dashboard_filename}`")
        st.info("Please make sure the HTML dashboard file is in the folder.")
        
    else:
        # 3.2: Attempt to Load Model (Handling Version Mismatch)
        with st.spinner(f"Loading {selected_model} weights and dashboard..."):
            try:
                # We load the model just to prove it works/is compatible
                loaded_model = joblib.load(model_filename)
                st.success(f"✅ **Success:** {selected_model} loaded successfully!")
                
                # 3.3: Render the HTML Dashboard
                st.subheader(f"Interactive Dashboard: {selected_model}")
                
                with open(dashboard_filename, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Render HTML safely
                components.html(html_content, height=1000, scrolling=True)
                
            except Exception as e:
                # Catch version mismatch errors (scikit-learn version issues)
                st.error("⚠️ **Model Loading Error**")
                st.warning(f"""
                The model file `{model_filename}` could not be loaded. 
                
                **Technical Error:** {str(e)}
                
                **Possible Fix:** Your local `scikit-learn` version might differ from the one used to train the model (Google Colab).
                Try running: `pip install --upgrade scikit-learn` in your terminal.
                """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Developed by Arshyan & Moiz | SEECS Batch '23")