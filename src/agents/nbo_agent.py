import os
import json
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / '.env')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def next_best_offer_agent(applicant: dict, prediction: dict) -> dict:
    """
    Evaluates denied applications and proposes a "Next Best Offer" (NBO).
    Returns a structured JSON response with the offer recommendation.
    """
    if prediction.get('decision') == 'APPROVED':
        return {
            "triggered": False,
            "offer_type": "none",
            "offer_text": "",
            "reasoning": "Applicant was approved; NBO not required.",
            "tokens_used": 0
        }

    prompt = f"""You are a consumer finance expert acting as the "Next Best Offer" agent for LendWise Bank.
A loan application was just DENIED. We want to salvage the customer relationship by offering an alternative product or action.

Applicant Profile:
- Requested Amount: ${applicant.get('loan_amnt', 0):,}
- Income: ${applicant.get('person_income', 0):,}
- Employment Length: {applicant.get('person_emp_exp', 0)} years
- Credit History Length: {applicant.get('cb_person_cred_hist_length', 0)} years
- Previous Defaults: {applicant.get('previous_loan_defaults_on_file', 'Unknown')}

Denial Context:
- Default Probability: {prediction.get('default_prob', 0):.0%}
- Top Risk Factors: {', '.join(prediction.get('top_factors', []))}

Your task: Determine the best alternative offer.
Choose ONE of the following offer types:
1. lower_amount (If income is steady but requested amount was too high)
2. secured_loan (If they have poor credit history or previous defaults)
3. financial_counseling_and_reapply (If profile is very risky)

You MUST respond ONLY with a valid JSON object. No markdown, no explanation. Just raw JSON:
{{
  "triggered": true,
  "offer_type": "lower_amount" OR "secured_loan" OR "financial_counseling_and_reapply",
  "offer_text": "One concise, empathetic sentence to easily append to the denial email proposing the alternative.",
  "reasoning": "One sentence internal reasoning for this choice."
}}"""

    response = client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.2,
        max_tokens  = 250
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps response
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            'triggered':       False,
            'offer_type':      'error',
            'offer_text':      '',
            'reasoning':       'Could not parse NBO JSON response.'
        }

    result['tokens_used'] = response.usage.total_tokens
    return result

if __name__ == '__main__':
    # Test 1 - Approved Profile (Should not trigger)
    good_app = {'loan_amnt': 5000, 'person_income': 90000, 'person_emp_exp': 5, 'cb_person_cred_hist_length': 10, 'previous_loan_defaults_on_file': 'No'}
    good_pred = {'decision': 'APPROVED', 'default_prob': 0.05, 'top_factors': ['credit_score']}
    
    # Test 2 - Risky requested too much
    high_amnt_app = {'loan_amnt': 30000, 'person_income': 40000, 'person_emp_exp': 2, 'cb_person_cred_hist_length': 2, 'previous_loan_defaults_on_file': 'No'}
    high_amnt_pred = {'decision': 'DENIED', 'default_prob': 0.65, 'top_factors': ['loan_to_income_pct']}
    
    # Test 3 - Previous defaults
    default_app = {'loan_amnt': 5000, 'person_income': 50000, 'person_emp_exp': 4, 'cb_person_cred_hist_length': 4, 'previous_loan_defaults_on_file': 'Yes'}
    default_pred = {'decision': 'DENIED', 'default_prob': 0.85, 'top_factors': ['previous_loan_defaults_on_file']}

    print("── Test 1: Approved Applicant (No NBO) ──")
    res1 = next_best_offer_agent(good_app, good_pred)
    for k, v in res1.items(): print(f"  {k:12s}: {v}")
    
    print("\n── Test 2: High Amount requested ──")
    res2 = next_best_offer_agent(high_amnt_app, high_amnt_pred)
    for k, v in res2.items(): print(f"  {k:12s}: {v}")
    
    print("\n── Test 3: Previous Default ──")
    res3 = next_best_offer_agent(default_app, default_pred)
    for k, v in res3.items(): print(f"  {k:12s}: {v}")
