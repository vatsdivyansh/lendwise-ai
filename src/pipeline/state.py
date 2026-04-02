from typing import Annotated, TypedDict, Dict, Any, List
import operator
from langchain_core.messages import BaseMessage

class LoanGraphState(TypedDict):
    """
    Master state schema for the loan approval LangGraph pipeline.
    This routes the applicant through ML prediction, Email Generation, 
    Guardrails, Bias analysis, and Next Best Offer determination.
    """
    
    # initial inputs ->
    applicant: dict
    
    # ml output ->
    ml_prediction: dict
    
    # email generation ->
    messages: Annotated[list[BaseMessage], operator.add]
    email_body: str
    email_validation_passed: bool
    email_tokens_used: int
    email_retries: int
    
    # guardrail ->
    guardrail_result: dict
    
    # bias detection
    bias_result: dict
    
    # next best offer 
    nbo_result: dict
    
    # final outcome 
    application_id: str
    final_status: str
