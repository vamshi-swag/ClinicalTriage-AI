import streamlit as st
from rag_engine import retrieve_medical_fact
from model_inference import generate_response

st.set_page_config(page_title="ClinicalTriage AI", page_icon="⚕️", layout="centered")

# Inject premium clinical-themed dark mode styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling with gradient */
    .clinical-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
    }
    
    .clinical-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Primary buttons with clinical gradients */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0px) !important;
    }
    
    /* Input fields */
    div[data-baseweb="textarea"] {
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
        transition: border-color 0.2s ease !important;
    }
    
    div[data-baseweb="textarea"]:focus-within {
        border-color: #38bdf8 !important;
    }
    
    textarea {
        color: #f1f5f9 !important;
        font-size: 1rem !important;
    }
    
    /* Output Card / Expander styling */
    div[data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 0.2rem !important;
    }
    
    /* Custom divider line */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="clinical-title">⚕️ ClinicalTriage AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="clinical-subtitle">Hybrid Fine-Tuned + RAG application codebase layout.</div>', unsafe_allow_html=True)

user_symptoms = st.text_area(
    "Enter Patient Symptoms or Case Notes:",
    height=120,
    placeholder="e.g., A 28-year-old female presents with sudden onset of lower right abdominal pain..."
)

if st.button("Analyze Patient Case", type="primary"):
    if user_symptoms.strip():
        with st.spinner("Processing local RAG extraction and optimizing model generation..."):
            try:
                # 1. RAG context query phase
                grounded_evidence = retrieve_medical_fact(user_symptoms)
                
                # 2. Synchronous hardware generation phase
                ai_triage_plan = generate_response(user_symptoms, grounded_evidence)
                
                # 3. Dynamic layout presentation
                st.subheader("📋 Triage Analysis Output")
                st.success(ai_triage_plan)
                
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                with st.expander("🔍 View Grounded RAG Data Context Injected"):
                    st.info(grounded_evidence)
                    
            except Exception as e:
                st.error(f"Execution Error occurred: {str(e)}")
                st.warning("Verify that your target workspace setup provides an active NVIDIA CUDA-capable framework.")
    else:
        st.warning("Please input case details prior to initiating model analysis.")
