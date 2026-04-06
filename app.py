import streamlit as st
import requests
import time
import PyPDF2

st.set_page_config(page_title="LendResponse AI", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    .stButton>button {
        width: 100%;
        background-color: #0f172a;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .decision-approved {
        background-color: #dcfce7;
        border-color: #bbf7d0;
    }
    .decision-approved p { color: #166534; }
    
    .decision-denied {
        background-color: #fee2e2;
        border-color: #fecaca;
    }
    .decision-denied p { color: #991b1b; }
</style>
""", unsafe_allow_html=True)

st.title("LendResponse AI")
st.markdown("Automated Underwriting System")
st.markdown("---")

API_URL_EXTRACT = "https://loan-approval-cg53.onrender.com/api/v1/extract"
API_URL_APPLY = "https://loan-approval-cg53.onrender.com/api/v1/apply"

# Session State Initializations
if "extracted_data" not in st.session_state: st.session_state.extracted_data = None
if "missing_fields" not in st.session_state: st.session_state.missing_fields = []
if "pipeline_results" not in st.session_state: st.session_state.pipeline_results = None
if "input_payload" not in st.session_state: st.session_state.input_payload = None

def stream_email(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

def run_main_pipeline(payload):
    st.session_state.input_payload = payload
    with st.spinner("Underwriting in progress..."):
        try:
            response = requests.post(API_URL_APPLY, json=payload)
            response.raise_for_status()
            st.session_state.pipeline_results = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the Backend API. Ensure 'uvicorn src.api:api --reload' is running in another terminal.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.subheader("Customer's Financial History")
    input_method = st.radio("Method", ["Upload Customer's Financial Document", "Enter Manually"], label_visibility="collapsed")
    st.markdown("---")
    
    if input_method == "Upload Customer's Financial Document":
        if st.session_state.extracted_data is None:
            uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])
            
            if st.button("Process Document"):
                document_content = ""
                if uploaded_file is not None:
                    if uploaded_file.name.endswith('.pdf'):
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            document_content += page.extract_text() + "\n"
                    else:
                        document_content = uploaded_file.getvalue().decode("utf-8")
                    
                    with st.spinner("Extracting parameters..."):
                        try:
                            sanitized_content = document_content.encode('utf-8', 'ignore').decode('utf-8')
                            res = requests.post(API_URL_EXTRACT, json={"document_text": sanitized_content})
                            res.raise_for_status()
                            st.session_state.extracted_data = res.json().get("extracted_data")
                            st.session_state.missing_fields = res.json().get("missing_fields")
                            st.session_state.pipeline_results = None
                            
                            if len(st.session_state.missing_fields) == 0:
                                run_main_pipeline(st.session_state.extracted_data)
                                st.rerun() 
                                
                        except Exception as e:
                            error_msg = str(e)
                            if hasattr(e, 'response') and e.response is not None:
                                try: error_msg = e.response.json().get('detail', str(e))
                                except: pass
                            st.error(f"Extraction failed: {error_msg}")
                else:
                    st.warning("Please upload a file first.")
        else:
            st.info("A document is currently being processed.")
            col_b1, col_b2 = st.columns(2)
            if col_b1.button("Upload Different Document", use_container_width=True):
                st.session_state.extracted_data = None
                st.session_state.missing_fields = []
                st.session_state.pipeline_results = None
                st.rerun()

        # HITL Interstitial State
        if st.session_state.extracted_data is not None and len(st.session_state.missing_fields) > 0 and st.session_state.pipeline_results is None:
            st.warning("⚠️ Document missing required details.")
            with st.form("hitl_form"):
                updates = {}
                missing = st.session_state.missing_fields
                col_h1, col_h2 = st.columns(2)
                
                with col_h1:
                    if "name" in missing: updates["name"] = st.text_input("Customer Name")
                    if "person_age" in missing: updates["person_age"] = st.number_input("Customer Age", min_value=18, max_value=100)
                    if "person_gender" in missing: updates["person_gender"] = st.selectbox("Customer Gender", ["female", "male", "non-binary"])
                    if "person_education" in missing: updates["person_education"] = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
                    if "person_income" in missing: updates["person_income"] = st.number_input("Annual Income ($)", min_value=1000.0)
                    if "person_emp_exp" in missing: updates["person_emp_exp"] = st.number_input("Employment Exp (Years)", min_value=0)
                    if "cb_person_cred_hist_length" in missing: updates["cb_person_cred_hist_length"] = st.number_input("Credit History Length (Years)", min_value=0)
                
                with col_h2:
                    if "person_home_ownership" in missing: updates["person_home_ownership"] = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
                    if "loan_amnt" in missing: updates["loan_amnt"] = st.number_input("Loan Amount ($)", min_value=100.0)
                    if "loan_intent" in missing: updates["loan_intent"] = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
                    if "loan_int_rate" in missing: updates["loan_int_rate"] = st.number_input("Interest Rate (%)", min_value=1.0)
                    if "credit_score" in missing: updates["credit_score"] = st.slider("FICO Score", min_value=300, max_value=850, value=650)
                    if "previous_loan_defaults_on_file" in missing: updates["previous_loan_defaults_on_file"] = st.selectbox("Previous Defaults?", ["No", "Yes"])
                
                submit_hitl = st.form_submit_button("Submit & Run Prediction")
                if submit_hitl:
                    extracted = st.session_state.extracted_data
                    for k, v in updates.items():
                        extracted[k] = v
                    if "loan_percent_income" in missing or "person_income" in updates or "loan_amnt" in updates:
                        inc = extracted.get("person_income", 1)
                        lamt = extracted.get("loan_amnt", 0)
                        extracted["loan_percent_income"] = round(lamt / inc, 2) if inc > 0 else 0.0

                    # Mark missing as 0 to exit HITL state UI lock
                    st.session_state.missing_fields = []
                    run_main_pipeline(extracted)
                    st.rerun()

    else:
        # Manual Entry Profile
        with st.form(key='manual_form'):
            m_name = st.text_input("Customer Name", value="", placeholder="Enter full name")
            col1, col2 = st.columns(2)
            with col1:
                m_age = st.number_input("Customer Age", min_value=18, max_value=100, value=None)
            with col2:
                m_gender = st.selectbox("Customer Gender", options=["female", "male", "non-binary"], index=None)
                
            m_education = st.selectbox("Education Level", options=["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=None)
            
            st.markdown("### Financials")
            m_income = st.number_input("Annual Income ($)", min_value=1000, value=None)
            m_emp_exp = st.number_input("Employment Exp (Years)", min_value=0, value=None)
            m_home_ownership = st.selectbox("Home Ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER"], index=None)
            
            st.markdown("### Loan Details")
            m_loan_amnt = st.number_input("Loan Amount ($)", min_value=100, value=None)
            m_loan_intent = st.selectbox("Loan Intent", options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=None)
            m_loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, value=None)
            
            st.markdown("### Credit Profile")
            m_credit_score = st.slider("FICO Score", min_value=300, max_value=850, value=650)
            m_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=None)
            m_defaults = st.selectbox("Previous Defaults?", options=["No", "Yes"], index=None)
            
            submit_manual = st.form_submit_button("Run Prediction")

        if submit_manual:
            fields = [m_name, m_age, m_gender, m_education, m_income, m_emp_exp, m_home_ownership, m_loan_amnt, m_loan_intent, m_loan_int_rate, m_cred_hist_length, m_defaults]
            if any(f is None or f == "" for f in fields):
                st.error("Please fill all required parameters.")
            else:
                dti = round(m_loan_amnt / m_income, 2) if m_income > 0 else 0.0
                payload = {
                    "name": m_name, "person_age": m_age, "person_gender": m_gender,
                    "person_education": m_education, "person_income": float(m_income),
                    "person_emp_exp": m_emp_exp, "person_home_ownership": m_home_ownership,
                    "loan_amnt": float(m_loan_amnt), "loan_intent": m_loan_intent,
                    "loan_int_rate": float(m_loan_int_rate), "loan_percent_income": dti,
                    "cb_person_cred_hist_length": m_cred_hist_length, "credit_score": m_credit_score,
                    "previous_loan_defaults_on_file": m_defaults
                }
                run_main_pipeline(payload)
                st.rerun()

# ── Right Dashboard: Outcomes & Output ──
with col_right:
    st.markdown("<h3 style='margin-top: -0.8rem; padding-bottom: 0.8rem;'>Prediction Dashboard</h3>", unsafe_allow_html=True)
    
    if st.session_state.pipeline_results is None:
        st.info("Awaiting application generation...")
    else:
        st.markdown("---")
        data = st.session_state.pipeline_results
        payload = st.session_state.input_payload
        decision = data.get('ml_decision')
        
        # Top line summary
        c1, c2, c3 = st.columns(3)
        
        # 1) Decision block
        with c1:
            if decision == 'APPROVED':
                st.markdown(f'<div class="metric-card decision-approved"><h3>Decision</h3><p>Eligible</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card decision-denied"><h3>Decision</h3><p>Denied</p></div>', unsafe_allow_html=True)
        
        # 2) Risk Tier
        with c2:
            st.markdown(f'<div class="metric-card"><h3>Risk Tier</h3><p>{data.get("ml_risk_tier")}</p></div>', unsafe_allow_html=True)
            
        # 3) Confidence
        with c3:
            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p>{data.get("ml_confidence")}</p></div>', unsafe_allow_html=True)
            
        
        # Highlights & Factors
        st.markdown("#### Application Highlights")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Annual Income", f"${payload.get('person_income', 0):,}")
        with h2:
            st.metric("Credit Score", f"{payload.get('credit_score', 0)}")
        with h3:
            st.metric("Debt-to-Income", f"{payload.get('loan_percent_income', 0):.0%}")
        
        st.markdown("#### Automated Reasoning")
        # Ensure we provide a very clean explanation of what tipped the model
        reasoning = f"The application was classified as **{data.get('ml_risk_tier')}**. "
        if decision == 'APPROVED':
            reasoning += "It passed underwriting minimums and the borrower shows sufficient capability to service the debt without exceedingly high default probability."
        else:
            reasoning += "The machine learning agent identified an unacceptable default probability."
        st.write(reasoning)

        # Email Generation Block
        st.markdown("---")
        st.markdown(f"#### Generated Customer Response (Email)")
        
        email_content = data.get('email_body', 'No email generated.')
        
        # Visual Streaming Effect via native Streamlit
        # We store streaming status so it doesn't replay on every button click elsewhere on the DOM
        stream_key = f"stream_{data.get('application_id')}"
        
        if stream_key not in st.session_state:
            with st.chat_message("assistant"):
                st.write_stream(stream_email(email_content))
            st.session_state[stream_key] = True
        else:
            with st.chat_message("assistant"):
                st.write(email_content)
