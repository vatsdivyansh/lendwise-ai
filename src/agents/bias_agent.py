import pandas as pd


def bias_detection_agent(decisions: list[dict]) -> dict:
    """
    Checks if approval rates differ significantly across gender groups.
    Uses the 80% rule (disparate impact ratio) — standard legal threshold.

    Input: list of dicts with keys 'person_gender' and 'decision'
    Example: [
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'female', 'decision': 'DENIED'},
    ]
    """
    if not decisions:
        return {
            'bias_detected':    False,
            'disparate_impact': None,
            'approval_rates':   {},
            'action':           'NONE',
            'reason':           'No decisions provided to analyze.'
        }

    df = pd.DataFrame(decisions)

    if 'person_gender' not in df.columns or 'decision' not in df.columns:
        return {
            'bias_detected':    False,
            'disparate_impact': None,
            'approval_rates':   {},
            'action':           'NONE',
            'reason':           'Missing required columns: person_gender, decision.'
        }

    # Approval rate per gender group
    approval_rates = (
        df.groupby('person_gender')['decision']
        .apply(lambda x: round((x == 'APPROVED').mean(), 4))
        .to_dict()
    )

    if len(approval_rates) < 2:
        return {
            'bias_detected':    False,
            'disparate_impact': None,
            'approval_rates':   approval_rates,
            'action':           'NONE',
            'reason':           'Need at least 2 gender groups to compare.'
        }

    majority_rate    = max(approval_rates.values())
    minority_rate    = min(approval_rates.values())
    disparate_impact = round(minority_rate / majority_rate, 4) if majority_rate > 0 else 1.0
    bias_detected    = disparate_impact < 0.80

    # Find which group has lower approval rate
    minority_group   = min(approval_rates, key=approval_rates.get)
    majority_group   = max(approval_rates, key=approval_rates.get)

    return {
        'bias_detected':    bias_detected,
        'disparate_impact': disparate_impact,
        'approval_rates':   approval_rates,
        'threshold':        0.80,
        'majority_group':   majority_group,
        'minority_group':   minority_group,
        'action':           'FLAG_FOR_HUMAN_REVIEW' if bias_detected else 'NONE',
        'reason': (
            f"Disparate impact {disparate_impact:.2f} is below 0.80 — "
            f"{minority_group} approval rate ({approval_rates[minority_group]:.0%}) is significantly "
            f"lower than {majority_group} ({approval_rates[majority_group]:.0%}). Potential bias detected."
            if bias_detected else
            f"Disparate impact {disparate_impact:.2f} is above 0.80 — "
            f"approval rates are within acceptable range across gender groups."
        )
    }


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Test 1 — No bias (rates are close)
    decisions_fair = [
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'male',   'decision': 'DENIED'},
        {'person_gender': 'female', 'decision': 'APPROVED'},
        {'person_gender': 'female', 'decision': 'APPROVED'},
        {'person_gender': 'female', 'decision': 'DENIED'},
    ]

    # Test 2 — Bias detected (female approval rate much lower)
    decisions_biased = [
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'male',   'decision': 'APPROVED'},
        {'person_gender': 'male',   'decision': 'DENIED'},
        {'person_gender': 'female', 'decision': 'APPROVED'},
        {'person_gender': 'female', 'decision': 'DENIED'},
        {'person_gender': 'female', 'decision': 'DENIED'},
        {'person_gender': 'female', 'decision': 'DENIED'},
    ]

    print("── Test 1: Fair decisions ──")
    result1 = bias_detection_agent(decisions_fair)
    for k, v in result1.items():
        print(f"  {k:20s}: {v}")

    print("\n── Test 2: Biased decisions ──")
    result2 = bias_detection_agent(decisions_biased)
    for k, v in result2.items():
        print(f"  {k:20s}: {v}")