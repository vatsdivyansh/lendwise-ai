import os
import sys

# Ensure Python can find the src module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.graph import app


def get_personas():
    """Returns predefined extreme applicant personas to demonstrate pipeline routing."""
    return {
        "1": {
            "title": "Super Safe Applicant",
            "applicant": {
                'name': 'Sarah Connor', 'person_age': 45, 'person_gender': 'female',
                'person_education': 'Doctorate', 'person_income': 180000, 'person_emp_exp': 15,
                'person_home_ownership': 'OWN', 'loan_amnt': 5000, 'loan_intent': 'HOMEIMPROVEMENT',
                'loan_int_rate': 6.5, 'loan_percent_income': 0.03, 'cb_person_cred_hist_length': 20,
                'credit_score': 810, 'previous_loan_defaults_on_file': 'No'
            }
        },
        "2": {
            "title": "Borderline NBO Trigger",
            "applicant": {
                'name': 'Jordan Smith', 'person_age': 28, 'person_gender': 'non-binary',
                'person_education': 'High School', 'person_income': 55000, 'person_emp_exp': 4,
                'person_home_ownership': 'RENT', 'loan_amnt': 25000, 'loan_intent': 'VENTURE',
                'loan_int_rate': 14.5, 'loan_percent_income': 0.45, 'cb_person_cred_hist_length': 6,
                'credit_score': 680, 'previous_loan_defaults_on_file': 'No'
            }
        },
        "3": {
            "title": "Red Flag Applicant",
            "applicant": {
                'name': 'Sam Rogers', 'person_age': 21, 'person_gender': 'male',
                'person_education': 'High School', 'person_income': 24000, 'person_emp_exp': 1,
                'person_home_ownership': 'RENT', 'loan_amnt': 15000, 'loan_intent': 'PERSONAL',
                'loan_int_rate': 18.2, 'loan_percent_income': 0.62, 'cb_person_cred_hist_length': 2,
                'credit_score': 540, 'previous_loan_defaults_on_file': 'Yes'
            }
        }
    }


def execute_pipeline(applicant: dict):
    """Streams the applicant through the LangGraph Master Pipeline."""
    print(f"\n🚀 Initializing Loan Processing for: {applicant['name']}...")
    
    initial_state = {
        "applicant": applicant,
        "email_retries": 0,
        "messages": []
    }
    
    # Run through the graph
    print("-" * 50)
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f" ✅ Executed Node -> [ {node_name.upper()} ]")
            
    # Retrieve final execution state
    final_state = app.invoke(initial_state)
    
    # Pretty print the final breakdown
    print("\n" + "=" * 60)
    print(" 📊 PIPELINE DECISION SUMMARY")
    print("=" * 60)
    print(f"ML Decision:          {final_state['ml_prediction']['decision']}")
    print(f"Final DB Status:      {final_state['final_status']}")
    print(f"MongoDB Record ID:    {final_state['application_id']}")
    
    nbo = final_state.get('nbo_result', {})
    if nbo.get('triggered'):
        print(f"NBO Triggered:        Yes ({nbo.get('offer_type')})")
        print(f"NBO Recommendation:   {nbo.get('reasoning')}")
    else:
        print("NBO Triggered:        No")
        
    print(f"Guardrail Verdict:    {final_state.get('guardrail_result', {}).get('recommendation')}")
    
    print("\n📩 FINAL COMMUNICATED EMAIL:")
    print("-" * 60)
    print(final_state['email_body'])
    print("-" * 60)


if __name__ == '__main__':
    personas = get_personas()
    
    print("\n🤖 Welcome to LendWise AI Loan System")
    print("Select a test persona to route through the agentic pipeline:\n")
    
    for key, data in personas.items():
        print(f" [{key}] {data['title']}")
    print(" [q] Quit\n")
    
    while True:
        choice = input("Enter your choice (1-3 or q): ").strip().lower()
        if choice == 'q':
            print("Exiting system...")
            sys.exit(0)
            
        if choice in personas:
            selected = personas[choice]
            print(f"\n>>> Loading Profile: {selected['title']}")
            try:
                execute_pipeline(selected["applicant"])
            except Exception as e:
                print(f"Pipeline crashed: {str(e)}")
            break
        else:
            print("Invalid choice, please select 1, 2, 3 or q.")
