import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from pipeline.state import LoanGraphState
from pipeline.nodes import (
    predict_node, email_node, guardrail_node, 
    nbo_node, bias_node, final_status_node
)

def route_after_prediction(state: LoanGraphState) -> list[str]:
    """
    Parallel routing after prediction.
    We can run the NBO evaluation, Email Generation, and Bias check concurrently 
    since they only strictly depend on the prediction result.
    Wait, LangGraph standard edges run sequentially if not returned in a list, 
    but let's do sequential with conditional skipping.
    """
    pass

# ── Assemble Graph ───────────────────────────────────────────────────────────
workflow = StateGraph(LoanGraphState)

# 1. Add all nodes
workflow.add_node("predict", predict_node)
workflow.add_node("email", email_node)
workflow.add_node("nbo", nbo_node)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("bias", bias_node)
workflow.add_node("finalize", final_status_node)


# 2. Define edge routing logic
def should_trigger_nbo(state: LoanGraphState) -> str:
    """Conditionally route to NBO only if the loan was denied."""
    decision = state['ml_prediction'].get('decision')
    if decision == 'DENIED':
        return "nbo"
    return "guardrail"


def verify_guardrail(state: LoanGraphState) -> str:
    """
    If guardrail explicitly says REVISE (which is a minor fix), we route back to email.
    If it says ESCALATE_TO_HUMAN, we proceed to final status to flag it.
    """
    recommendation = state['guardrail_result'].get('recommendation')
    if recommendation == 'REVISE':
        # Route back to email generation to try again
        # Note: email node needs to support handling this, but for now we loop it.
        # email_generator natively loops via its own subgraph, but if it escapes 
        # and fails the enterprise guardrail, we force a complete redraft.
        return "email"
    return "bias"


# 3. Connect Graph Edges
workflow.add_edge(START, "predict")
workflow.add_edge("predict", "email")

# After email is generated, check if we need an NBO appended
workflow.add_conditional_edges(
    "email",
    should_trigger_nbo,
    {
        "nbo": "nbo",
        "guardrail": "guardrail"
    }
)
workflow.add_edge("nbo", "guardrail")

# Guardrail conditional edge
workflow.add_conditional_edges(
    "guardrail",
    verify_guardrail,
    {
        "email": "email",
        "bias": "bias"
    }
)

workflow.add_edge("bias", "finalize")
workflow.add_edge("finalize", END)


# 4. Compile
app = workflow.compile()


# Testing
if __name__ == '__main__':
    # Test pipeline with a risky applicant
    test_applicant = {
        'name': 'Sam Rogers',
        'person_age': 21,
        'person_gender': 'male',
        'person_education': 'High School',
        'person_income': 24000,
        'person_emp_exp': 1,
        'person_home_ownership': 'RENT',
        'loan_amnt': 15000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 18.2,
        'loan_percent_income': 0.62,
        'cb_person_cred_hist_length': 2,
        'credit_score': 540,
        'previous_loan_defaults_on_file': 'Yes'
    }

    print("Triggering Master Pipeline...")
    
    # Initialize state
    initial_state = {
        "applicant": test_applicant,
        "email_retries": 0,
        "messages": []
    }
    
    # stream through the nodes
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"Finished Node: {node_name}")
            
    # Final Result
    final_state = app.invoke(initial_state)
    print("\n" + "="*50)
    print(" MASTER PIPELINE RESULT:")
    print(f"Decision: {final_state['ml_prediction']['decision']}")
    print(f"Final DB Status: {final_state['final_status']}")
    print(f"App ID: {final_state['application_id']}")
    print(f"NBO Triggered: {final_state.get('nbo_result', {}).get('triggered', False)}")
    print(f"Guardrail Recommendation: {final_state.get('guardrail_result', {}).get('recommendation')}")
    print("\nFINAL EMAIL:")
    print(final_state['email_body'])
