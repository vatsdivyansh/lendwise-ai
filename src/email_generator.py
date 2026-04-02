import os
import operator
from typing import Annotated, TypedDict, Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Define state typed dict
class EmailState(TypedDict):
    applicant: dict
    prediction: dict
    decision_context: str
    messages: Annotated[list, operator.add]
    email_body: str
    validation: dict
    tokens_used: int
    retries: int

def get_decision_context(applicant: dict, prediction: dict) -> str:
    decision     = prediction['decision']
    approval_prob = prediction['approval_prob']
    default_prob  = prediction['default_prob']
    top_factors  = prediction['top_factors']
    risk_tier    = prediction['risk_tier']
    
    factor_map = {
        'previous_loan_defaults_on_file': 'previous loan default history',
        'loan_to_income_pct':             'loan amount relative to income',
        'person_home_ownership_RENT':     'current rental housing status',
        'person_home_ownership_OWN':      'home ownership status',
        'credit_score':                   'credit score',
        'debt_to_income':                 'debt-to-income ratio',
        'rate_times_amnt':                'total interest burden',
        'log_income':                     'annual income level',
        'person_emp_exp':                 'employment experience',
        'cb_person_cred_hist_length':     'credit history length',
    }
    explained_factors = [
        factor_map.get(f, f.replace('_', ' ')) for f in top_factors
    ]

    if decision == 'APPROVED':
        return f"""The loan has been APPROVED.
- Approval confidence: {approval_prob:.0%}
- Risk tier: {risk_tier}
- Key positive factors: {', '.join(explained_factors)}

Write a warm, professional approval email. Include:
1. Clear approval statement
2. Mention the loan amount and purpose
3. Brief note on what made this application strong
4. Next steps (document submission, expect call within 2 business days)
5. Contact info placeholder: support@lendwise.com"""
    else:
        return f"""The loan has been DENIED.
- Default risk: {default_prob:.0%}
- Risk tier: {risk_tier}
- Key concern factors: {', '.join(explained_factors)}

Write an empathetic, professional denial email. Include:
1. Clear but sensitive denial statement
2. Mention the loan amount and purpose
3. Top 2 reasons explained in simple non-technical language
4. 2-3 concrete steps the applicant can take to improve their profile
5. Offer to reapply in 6 months
6. Contact info placeholder: support@lendwise.com"""


# ── Nodes ────────────────────────────────────────────────────────────────────

def prepare_node(state: EmailState):
    """Prepares the initial prompts."""
    applicant = state['applicant']
    prediction = state['prediction']
    
    name         = applicant.get('name', 'Applicant')
    loan_amnt    = applicant.get('loan_amnt', 'N/A')
    loan_intent  = applicant.get('loan_intent', 'N/A').title()
    
    decision_context = get_decision_context(applicant, prediction)
    
    system_prompt = """You are a professional loan officer at LendWise Bank.
Write clear, empathetic, and legally compliant email responses to loan applicants.
Rules:
- Never mention specific interest rates or terms not provided to you
- Never make false promises about future approvals
- Keep emails under 250 words
- Use plain English, avoid financial jargon
- Always be respectful and encouraging regardless of decision
- Do not include a subject line, just the email body"""

    user_prompt = f"""Write a loan decision email for:
Applicant name: {name}
Loan amount requested: ${loan_amnt:,}
Loan purpose: {loan_intent}

Decision details:
{decision_context}"""

    # Start fresh if no messages
    if not state.get('messages'):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    else:
        messages = []
        
    return {
        "decision_context": decision_context,
        "messages": messages
    }

def generate_email_node(state: EmailState):
    """Calls ChatGroq to generate the email."""
    # Ensure client is instantiated per execution to grab the latest env vars
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, max_tokens=500)
    
    # We pass all messages to the LLM (which includes the system prompt + user prompts + potential retries)
    response = llm.invoke(state['messages'])
    
    email_body = response.content.strip()
    
    # Safely get tokens
    tokens = 0
    if response.response_metadata and 'token_usage' in response.response_metadata:
        tokens = response.response_metadata['token_usage'].get('total_tokens', 0)
    
    return {
        "email_body": email_body,
        "messages": [response],
        "tokens_used": state.get('tokens_used', 0) + tokens
    }

def validate_email_node(state: EmailState):
    """Validates the generated email."""
    email_body = state['email_body']
    decision = state['prediction']['decision']
    
    checks = {
        'has_decision_word': (
            decision.lower() in email_body.lower() or
            ('congratulations' in email_body.lower() if decision == 'APPROVED'
             else 'unable' in email_body.lower() or 'unfortunately' in email_body.lower())
        ),
        'acceptable_length': 50 <= len(email_body.split()) <= 300,
        'has_contact_info':  'lendwise' in email_body.lower(),
        'no_fake_rates':     'interest rate of' not in email_body.lower(),
    }
    
    validation_passed = all(checks.values())
    
    validation_result = {
        'passed': validation_passed,
        'checks': checks
    }
    
    return {
        "validation": validation_result,
        "retries": state.get('retries', 0) + 1
    }

def should_retry(state: EmailState) -> str:
    """Routing function to determine if we should retry or stop."""
    validation = state['validation']
    retries = state.get('retries', 0)
    
    if validation['passed'] or retries >= 3:
        return END
    else:
        return "prepare_retry"

def prepare_retry_node(state: EmailState):
    """If validation fails, we append a human message with the errors."""
    failed_checks = [k for k, v in state['validation']['checks'].items() if not v]
    correction_prompt = f"Your previous email failed validation on these checks: {', '.join(failed_checks)}. Please revise the email to address these issues, keep it under 250 words, and do not include fake interest rates or a subject line."
    return {"messages": [HumanMessage(content=correction_prompt)]}


# ── Compile Graph ────────────────────────────────────────────────────────────

workflow = StateGraph(EmailState)

workflow.add_node("prepare", prepare_node)
workflow.add_node("generate", generate_email_node)
workflow.add_node("validate", validate_email_node)
workflow.add_node("prepare_retry", prepare_retry_node)

workflow.add_edge(START, "prepare")
workflow.add_edge("prepare", "generate")
workflow.add_edge("generate", "validate")

workflow.add_conditional_edges(
    "validate",
    should_retry,
    {
        END: END,
        "prepare_retry": "prepare_retry"
    }
)
workflow.add_edge("prepare_retry", "generate")

graph = workflow.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def generate_email(applicant: dict, prediction: dict) -> dict:
    """
    Takes raw applicant data + prediction result,
    returns a professionally written loan decision email using LangGraph.
    """
    initial_state = {
        "applicant": applicant,
        "prediction": prediction,
        "tokens_used": 0,
        "retries": 0,
        "messages": []
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return {
        'decision':   prediction['decision'],
        'email_body': final_state['email_body'],
        'tokens_used': final_state['tokens_used'],
        'validation_passed': final_state['validation']['passed'],
        'checks': final_state['validation']['checks']
    }

def validate_email(email_body: str, decision: str) -> dict:
    """Maintained for backwards compatibility if called standalone."""
    checks = {
        'has_decision_word': (
            decision.lower() in email_body.lower() or
            ('congratulations' in email_body.lower() if decision == 'APPROVED'
             else 'unable' in email_body.lower() or 'unfortunately' in email_body.lower())
        ),
        'acceptable_length': 50 <= len(email_body.split()) <= 300,
        'has_contact_info':  'lendwise' in email_body.lower(),
        'no_fake_rates':     'interest rate of' not in email_body.lower(),
    }
    return {
        'passed': all(checks.values()),
        'checks': checks
    }

# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from predict import predict_single

    # Test applicant 1 - strong profile
    applicant_good = {
        'name':                          'Rahul Sharma',
        'person_age':                    35,
        'person_gender':                 'male',
        'person_education':              'Master',
        'person_income':                 85000,
        'person_emp_exp':                8,
        'person_home_ownership':         'OWN',
        'loan_amnt':                     10000,
        'loan_intent':                   'PERSONAL',
        'loan_int_rate':                 8.5,
        'loan_percent_income':           0.12,
        'cb_person_cred_hist_length':    10,
        'credit_score':                  720,
        'previous_loan_defaults_on_file':'No'
    }

    # Test applicant 2 - risky profile
    applicant_risky = {
        'name':                          'Priya Mehta',
        'person_age':                    22,
        'person_gender':                 'female',
        'person_education':              'High School',
        'person_income':                 22000,
        'person_emp_exp':                0,
        'person_home_ownership':         'RENT',
        'loan_amnt':                     15000,
        'loan_intent':                   'VENTURE',
        'loan_int_rate':                 18.5,
        'loan_percent_income':           0.68,
        'cb_person_cred_hist_length':    1,
        'credit_score':                  480,
        'previous_loan_defaults_on_file':'Yes'
    }

    for applicant in [applicant_good, applicant_risky]:
        print(f"\n{'='*60}")
        print(f"Applicant: {applicant['name']}")

        prediction = predict_single(applicant)
        print(f"ML Decision: {prediction['decision']} "
              f"(approval: {prediction['approval_prob']:.0%})")

        result = generate_email(applicant, prediction)

        print(f"Validation passed: {result['validation_passed']}")
        print(f"Checks: {result['checks']}")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"\n── Generated Email ──\n")
        print(result['email_body'])