Live URL:https://lendwise-ai.streamlit.app/
Render API: https://loan-approval-cg53.onrender.com/

LendResponse AI:
An automated loan underwriting system that takes a borrower's financial document, extracts the relevant data using an LLM, runs it through a trained XGBoost credit risk model, and generates a compliant decision email — all in a single pipeline.

What I Built:
Traditional loan underwriting is slow and inconsistent. A bank officer reads a document, manually fills a risk form, runs a credit check, and drafts a decision letter — repeated hundreds of times a week. I built a system that automates this entire workflow end-to-end.
A bank officer uploads a PDF (W2, pay stub, bank statement). A Llama-3.3-70B model extracts the financial fields from the raw text into a validated schema. If any fields are missing, the UI surfaces only those fields for the officer to fill in manually — the Human-in-the-Loop step. The completed application is then submitted to a LangGraph pipeline that runs four steps in sequence: an XGBoost classifier predicts the probability of default, an LLM generates a personalised approval or denial email, a compliance agent audits the email for hallucinations and discriminatory language, and a bias detection agent checks whether approval rates across gender groups have drifted below the legal 80% threshold. The full decision journey — input, ML output, email, guardrail verdict, bias check — is serialised to MongoDB for regulatory audit.
Denied applicants are not simply rejected. A Next Best Offer agent analyses the denial reasons and proposes an alternative: a lower loan amount, a secured loan, or a path to reapply after credit improvement.

Tech Stack

XGBoost — credit risk classification, trained on ~58K historical loan records
LangGraph — state machine orchestration for the multi-step AI pipeline
LangChain + Groq (Llama-3.3-70B) — document extraction and email generation
FastAPI — REST API backend with Pydantic validation
Streamlit — bank officer UI with document upload and HITL form rendering
MongoDB — audit trail storage for every application decision
scikit-learn / pandas / NumPy — preprocessing and feature engineering

![alt text](<Screenshot 2026-04-06 103715.png>)

