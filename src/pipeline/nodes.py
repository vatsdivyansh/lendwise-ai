import os
import sys

# Ensure we can import from src directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.state import LoanGraphState

# Import core functions
from predict import predict_single
from email_generator import generate_email
from agents.guardrail_agent import content_guardrail_agent
from agents.nbo_agent import next_best_offer_agent
from agents.bias_agent import bias_detection_agent
from database import get_recent_applications, save_application

def predict_node(state: LoanGraphState) -> dict:
    """Runs the machine learning prediction model."""
    applicant = state['applicant']
    prediction = predict_single(applicant)
    
    return {"ml_prediction": prediction}


def email_node(state: LoanGraphState) -> dict:
    """Generates the email leveraging the isolated email generator subgraph."""
    applicant = state['applicant']
    prediction = state['ml_prediction']
    
    # generate_email internally handles retries and LangChain LLM logic
    result = generate_email(applicant, prediction)
    
    return {
        "email_body": result['email_body'],
        "email_tokens_used": result['tokens_used'],
        "email_validation_passed": result['validation_passed']
    }


def guardrail_node(state: LoanGraphState) -> dict:
    """Evaluates the generated email for compliance issues."""
    email_body = state['email_body']
    decision = state['ml_prediction']['decision']
    
    result = content_guardrail_agent(email_body, decision)
    
    return {"guardrail_result": result}


def nbo_node(state: LoanGraphState) -> dict:
    """Determines the Next Best Offer if the loan was denied."""
    applicant = state['applicant']
    prediction = state['ml_prediction']
    
    result = next_best_offer_agent(applicant, prediction)
    
    return {"nbo_result": result}


def bias_node(state: LoanGraphState) -> dict:
    """Checks for disparate impact against protected classes across recent decisions."""
    current_applicant_data = {
        'person_gender': state['applicant'].get('person_gender', 'unknown'),
        'decision': state['ml_prediction']['decision']
    }
    
    # Fetch recent decisions from DB to calculate aggregate bias
    recent_apps = get_recent_applications(limit=49)
    past_decisions = []
    for app in recent_apps:
        past_decisions.append({
            'person_gender': app.get('applicant', {}).get('person_gender', 'unknown'),
            'decision': app.get('ml_prediction', {}).get('decision', 'UNKNOWN')
        })
        
    # Append the current decision
    all_decisions = past_decisions + [current_applicant_data]
    
    result = bias_detection_agent(all_decisions)
    
    return {"bias_result": result}


def final_status_node(state: LoanGraphState) -> dict:
    """Calculates final status and persists to the database."""
    # Determine the final status based on all agent evaluations
    guardrail = state.get('guardrail_result', {})
    bias = state.get('bias_result', {})
    
    if guardrail.get('recommendation') in ['REVISE', 'ESCALATE_TO_HUMAN']:
        final_status = "FLAGGED_GUARDRAIL"
    elif bias.get('action') == 'FLAG_FOR_HUMAN_REVIEW':
        final_status = "FLAGGED_BIAS"
    else:
        final_status = "EMAIL_READY"
        
    # Save the complete object journey
    app_id = save_application(
        applicant=state['applicant'],
        prediction=state['ml_prediction'],
        email_result={"email_body": state.get('email_body'), "tokens_used": state.get('email_tokens_used')},
        guardrail_result=guardrail,
        bias_result=bias,
        nbo_result=state.get('nbo_result', {}),
        final_status=final_status
    )
    
    return {
        "final_status": final_status,
        "application_id": app_id
    }
