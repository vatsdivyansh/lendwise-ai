import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.pipeline.graph import app as graph_pipeline
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Ensure env vars are loaded for API
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Configure the API
api = FastAPI(
    title="LendWise AI API",
    description="Backend AI API powering the full-stack loan approval system.",
    version="1.0.0"
)

class DocumentExtractionRequest(BaseModel):
    document_text: str

# Schema for AI extraction where fields can be None if not found
class PartialApplicantSchema(BaseModel):
    name: Optional[str] = Field(None, description="Applicant's full name")
    person_age: Optional[int] = Field(None, description="Age in years. Extract if available.")
    person_gender: Optional[str] = Field(None, description="Gender: male, female, or non-binary")
    person_education: Optional[str] = Field(None, description="Highest degree. E.g., High School, Bachelor, Master")
    person_income: Optional[float] = Field(None, description="Annual income in USD")
    person_emp_exp: Optional[int] = Field(None, description="Years of employment experience")
    person_home_ownership: Optional[str] = Field(None, description="Housing status: RENT, OWN, MORTGAGE, OTHER")
    loan_amnt: Optional[float] = Field(None, description="Requested loan amount in USD")
    loan_intent: Optional[str] = Field(None, description="Purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_int_rate: Optional[float] = Field(None, description="Proposed interest rate percentage")
    loan_percent_income: Optional[float] = Field(None, description="Loan amount relative to income")
    cb_person_cred_hist_length: Optional[int] = Field(None, description="Years since first active credit line")
    credit_score: Optional[int] = Field(None, description="FICO score")
    previous_loan_defaults_on_file: Optional[str] = Field(None, description="'Yes' or 'No'")

@api.post("/api/v1/extract")
async def extract_document(request: DocumentExtractionRequest):
    """
    Agentic document extraction endpoint.
    Accepts raw financial document text, uses ChatGroq to map it to a structured JSON.
    """
    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        structured_llm = llm.with_structured_output(PartialApplicantSchema)
        
        prompt = f"""You are an expert financial document analyzer at a bank.
Extract the relevant customer application metrics from the text below. 
If a field is not explicitly mentioned or clearly determinable, do NOT guess. Set it to null.
For 'previous_loan_defaults_on_file', respond 'Yes' or 'No' if known.

Document Content:
{request.document_text}
"""
        extracted = structured_llm.invoke(prompt)
        
        # Guard against None when LLM fails to parse structure
        if extracted is None:
            extracted_dict = {k: None for k in PartialApplicantSchema.model_fields.keys()}
        elif hasattr(extracted, "model_dump"):
            extracted_dict = extracted.model_dump()
        else:
            extracted_dict = extracted
        
        # Determine missing fields
        missing_fields = [k for k, v in extracted_dict.items() if v is None]
        
        return {
            "status": "success",
            "extracted_data": extracted_dict,
            "missing_fields": missing_fields
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the strict data validation schema
class ApplicantSchema(BaseModel):
    name: str = Field(..., description="Applicant's full name")
    person_age: int = Field(..., ge=18, description="Age in years")
    person_gender: str = Field(..., description="Gender: male, female, or non-binary")
    person_education: str = Field(..., description="Highest degree. E.g., High School, Bachelor, Master")
    person_income: float = Field(..., ge=0, description="Annual income in USD")
    person_emp_exp: int = Field(..., ge=0, description="Years of employment experience")
    person_home_ownership: str = Field(..., description="Housing status: RENT, OWN, MORTGAGE, OTHER")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_intent: str = Field(..., description="Purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_int_rate: float = Field(..., ge=0, description="Proposed interest rate percentage")
    loan_percent_income: float = Field(..., ge=0, le=1.0, description="Loan amount relative to income")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Years since first active credit line")
    credit_score: int = Field(..., ge=300, le=850, description="FICO score")
    previous_loan_defaults_on_file: str = Field(..., description="'Yes' or 'No'")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Sarah Connor",
                "person_age": 45,
                "person_gender": "female",
                "person_education": "Doctorate",
                "person_income": 180000,
                "person_emp_exp": 15,
                "person_home_ownership": "OWN",
                "loan_amnt": 5000,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_int_rate": 6.5,
                "loan_percent_income": 0.03,
                "cb_person_cred_hist_length": 20,
                "credit_score": 810,
                "previous_loan_defaults_on_file": "No"
            }
        }
    }

@api.post("/api/v1/apply")
async def evaluate_application(applicant: ApplicantSchema):

    """
    Submits an applicant profile to the LangGraph AI Pipeline.
    Validates input -> Queries ML -> Routes Email Generation -> Audits Guardrails -> Checks NBO -> Returns Final Package.
    """
    try:
        # Convert strict Pydantic model to dict for the pipeline
        applicant_dict = applicant.model_dump()
        
        # Initialize the LangGraph state
        initial_state = {
            "applicant": applicant_dict,
            "email_retries": 0,
            "messages": []
        }
        
        # Fire the graph synchronously 
        final_state = graph_pipeline.invoke(initial_state)
        
        # Format a clean response payload for the UI
        return {
            "status": "success",
            "application_id": final_state.get('application_id'),
            "database_flag": final_state.get('final_status'),
            "ml_decision": final_state['ml_prediction'].get('decision'),
            "ml_risk_tier": final_state['ml_prediction'].get('risk_tier'),
            "ml_confidence": final_state['ml_prediction'].get('confidence'),
            "nbo_triggered": final_state.get('nbo_result', {}).get('triggered', False),
            "nbo_recommendation": final_state.get('nbo_result', {}).get('offer_type', 'none'),
            "guardrail_verdict": final_state.get('guardrail_result', {}).get('recommendation'),
            "email_body": final_state.get('email_body')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
