import os
import json
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / '.env')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def content_guardrail_agent(email_body: str, decision: str) -> dict:
    """
    Uses Llama to judge if the generated email is safe to send.
    Checks for: hallucinations, discrimination, false promises,
    unprofessional tone, missing decision statement.

    Returns a structured verdict with recommendation:
    - SEND           → email is clean, send it
    - REVISE         → minor issues, regenerate email
    - ESCALATE_TO_HUMAN → serious issues, human must review
    """
    prompt = f"""You are a strict compliance officer at a bank reviewing an auto-generated loan decision email before it is sent to a customer.

Analyze this email carefully for ALL of the following issues:

1. HALLUCINATION — Does it mention specific interest rates, repayment schedules, or policy details that were not confirmed in the decision?
2. DISCRIMINATION — Does it contain any language that could be discriminatory based on gender, age, race, religion, or nationality?
3. FALSE_PROMISE — Does it guarantee future approvals, specific rates, or outcomes that cannot be confirmed?
4. UNPROFESSIONAL — Is the tone condescending, overly casual, or inappropriate for a bank communication?
5. MISSING_DECISION — Is the loan decision (approved or denied) clearly and unambiguously stated?
6. FACTUAL_ERROR — Does the email contradict the actual decision provided?

Email to review:
\"\"\"{email_body}\"\"\"

Expected decision: {decision}

You MUST respond ONLY with a valid JSON object. No explanation, no markdown, no code fences. Just raw JSON:
{{
  "safe_to_send": true or false,
  "issues_found": ["list each issue type found, e.g. HALLUCINATION, DISCRIMINATION — empty list if none"],
  "severity": "none" or "low" or "medium" or "high",
  "recommendation": "SEND" or "REVISE" or "ESCALATE_TO_HUMAN",
  "explanation": "one sentence explaining your verdict"
}}"""

    response = client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0,
        max_tokens  = 400
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
            'safe_to_send':    False,
            'issues_found':    ['PARSE_ERROR'],
            'severity':        'high',
            'recommendation':  'ESCALATE_TO_HUMAN',
            'explanation':     'Could not parse guardrail response — escalating to human.'
        }

    result['tokens_used'] = response.usage.total_tokens
    return result


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Test 1 — Clean approval email
    clean_email = """Dear Rahul Sharma,

We are pleased to inform you that your personal loan application 
for $10,000 has been approved. Based on your strong financial profile, 
including your excellent credit history and stable income, we are 
confident in your ability to manage this loan.

Please expect a call from our team within 2 business days to guide 
you through the next steps, including document submission.

If you have any questions, please contact us at support@lendwise.com.

Best regards,
LendWise Bank Team"""

    # Test 2 — Problematic email with hallucinated rate and false promise
    problematic_email = """Dear Priya Mehta,

Unfortunately we cannot approve your loan at this time. However, 
if you reapply next month we guarantee you will be approved at 
an interest rate of 5.9%. Your application was rejected because 
of your age and where you are from.

Contact us at support@lendwise.com.

Regards,
LendWise"""

    print("── Test 1: Clean email ──")
    result1 = content_guardrail_agent(clean_email, "APPROVED")
    for k, v in result1.items():
        print(f"  {k:15s}: {v}")

    print("\n── Test 2: Problematic email ──")
    result2 = content_guardrail_agent(problematic_email, "DENIED")
    for k, v in result2.items():
        print(f"  {k:15s}: {v}")