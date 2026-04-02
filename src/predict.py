import pandas as pd
import numpy as np
import joblib
from preprocess import load_and_preprocess



import os
# ── Load model once ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model       = joblib.load(os.path.join(BASE_DIR, 'models', 'loan_model_v1.pkl'))
FEATURE_COLS = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'))


def predict_single(applicant: dict) -> dict:
    """
    Takes raw applicant data as a dict, returns decision + confidence.

    Example input:
    {
        'person_age': 30,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 60000,
        'person_emp_exp': 5,
        'person_home_ownership': 'RENT',
        'loan_amnt': 10000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.17,
        'cb_person_cred_hist_length': 4,
        'credit_score': 650,
        'previous_loan_defaults_on_file': 'No'
    }
    """
    # Convert to single-row DataFrame
    df = pd.DataFrame([applicant])

    # Add dummy loan_status column so preprocess doesn't break
    df['loan_status'] = 0

    # Reuse exact same preprocessing pipeline
    df = load_and_preprocess_single(df)

    # Align columns to what model was trained on
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    # NEW - prob is probability of DEFAULT (1=default, 0=approved)
    default_prob  = model.predict_proba(df)[0][1]
    approval_prob = 1 - default_prob
    decision      = 'APPROVED' if approval_prob >= 0.6 else 'DENIED'
    confidence    = approval_prob if decision == 'APPROVED' else default_prob

    # Top contributing features
    importance  = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_factors = sorted(importance, key=importance.get, reverse=True)[:3]

    return {
    'decision':       decision,
    'approval_prob':  round(float(approval_prob), 4),
    'default_prob':   round(float(default_prob), 4),
    'confidence':     f"{confidence:.0%}",
    'top_factors':    top_factors,
    'risk_tier':      _risk_tier(approval_prob)
}


def load_and_preprocess_single(df):
    """Mirrors preprocess.py logic for a single row."""
    df = df.copy()

    # Binary encodings
    df['person_gender'] = (df['person_gender'] == 'male').astype(int)
    df['previous_loan_defaults_on_file'] = (
        df['previous_loan_defaults_on_file'] == 'Yes'
    ).astype(int)

    # Ordinal education
    edu_order = {
        'High School': 0, 'Associate': 1,
        'Bachelor': 2,   'Master': 3, 'Doctorate': 4
    }
    df['person_education'] = df['person_education'].map(edu_order)

    # Feature engineering
    df['debt_to_income']      = df['loan_amnt'] / (df['person_income'] + 1)
    df['loan_to_income_pct']  = df['loan_percent_income']
    df['income_per_emp_year'] = df['person_income'] / (df['person_emp_exp'] + 1)
    df['log_income']          = np.log(df['person_income'] + 1)
    df['log_loan_amnt']       = np.log(df['loan_amnt'] + 1)
    df['rate_times_amnt']     = df['loan_int_rate'] * df['loan_amnt']
    df['credit_per_age']      = df['credit_score'] / (df['person_age'] + 1)

    # Manual one-hot for home_ownership (drop_first=MORTGAGE is reference)
    ownership = df['person_home_ownership'].iloc[0]
    df['person_home_ownership_OTHER'] = int(ownership == 'OTHER')
    df['person_home_ownership_OWN']   = int(ownership == 'OWN')
    df['person_home_ownership_RENT']  = int(ownership == 'RENT')

    # Manual one-hot for loan_intent (drop_first=DEBTCONSOLIDATION is reference)
    intent = df['loan_intent'].iloc[0]
    df['loan_intent_EDUCATION']       = int(intent == 'EDUCATION')
    df['loan_intent_HOMEIMPROVEMENT'] = int(intent == 'HOMEIMPROVEMENT')
    df['loan_intent_MEDICAL']         = int(intent == 'MEDICAL')
    df['loan_intent_PERSONAL']        = int(intent == 'PERSONAL')
    df['loan_intent_VENTURE']         = int(intent == 'VENTURE')

    # Drop raw columns
    df.drop(['person_income', 'loan_amnt', 'loan_status',
             'person_home_ownership', 'loan_intent'], axis=1, inplace=True)

    return df


def _risk_tier(prob):
    if prob >= 0.75:   return 'Low Risk'
    elif prob >= 0.50: return 'Medium Risk'
    elif prob >= 0.25: return 'High Risk'
    else:              return 'Very High Risk'


# ── Test with two applicants ─────────────────────────────────────────────────
if __name__ == '__main__':

    applicant_good = {
        'person_age': 35,
        'person_gender': 'male',
        'person_education': 'Master',
        'person_income': 85000,
        'person_emp_exp': 8,
        'person_home_ownership': 'OWN',
        'loan_amnt': 10000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 8.5,
        'loan_percent_income': 0.12,
        'cb_person_cred_hist_length': 10,
        'credit_score': 720,
        'previous_loan_defaults_on_file': 'No'
    }

    applicant_risky = {
        'person_age': 22,
        'person_gender': 'female',
        'person_education': 'High School',
        'person_income': 22000,
        'person_emp_exp': 0,
        'person_home_ownership': 'RENT',
        'loan_amnt': 15000,
        'loan_intent': 'VENTURE',
        'loan_int_rate': 18.5,
        'loan_percent_income': 0.68,
        'cb_person_cred_hist_length': 1,
        'credit_score': 480,
        'previous_loan_defaults_on_file': 'Yes'
    }

    print("── Applicant 1 (Strong Profile) ──")
    result1 = predict_single(applicant_good)
    for k, v in result1.items():
        print(f"  {k:12s}: {v}")

    print("\n── Applicant 2 (Risky Profile) ──")
    result2 = predict_single(applicant_risky)
    for k, v in result2.items():
        print(f"  {k:12s}: {v}")



# if __name__ == '__main__':

#     # ── DIAGNOSTIC: check column alignment ──────────────────────────────
#     import json

#     applicant_good = {
#         'person_age': 35,
#         'person_gender': 'male',
#         'person_education': 'Master',
#         'person_income': 85000,
#         'person_emp_exp': 8,
#         'person_home_ownership': 'OWN',
#         'loan_amnt': 10000,
#         'loan_intent': 'PERSONAL',
#         'loan_int_rate': 8.5,
#         'loan_percent_income': 0.12,
#         'cb_person_cred_hist_length': 10,
#         'credit_score': 720,
#         'previous_loan_defaults_on_file': 'No',
#         'loan_status': 0
#     }

#     df = pd.DataFrame([applicant_good])
#     df = load_and_preprocess_single(df)
#     df = df.reindex(columns=FEATURE_COLS, fill_value=0)

#     print("── Column alignment check ──")
#     for col in FEATURE_COLS:
#         print(f"  {col:40s} = {df[col].iloc[0]}")

#     print(f"\nFeature cols count : {len(FEATURE_COLS)}")
#     print(f"Processed df cols  : {df.shape[1]}")        