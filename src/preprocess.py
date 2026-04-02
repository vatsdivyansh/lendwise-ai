import pandas as pd
import numpy as np


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # --- No missing values, so we go straight to feature engineering ---

    # 1. Engineered features
    df['debt_to_income']      = df['loan_amnt'] / (df['person_income'] + 1)
    df['loan_to_income_pct']  = df['loan_percent_income']                      # already exists, keep it
    df['income_per_emp_year'] = df['person_income'] / (df['person_emp_exp'] + 1)
    df['log_income']          = np.log(df['person_income'] + 1)
    df['log_loan_amnt']       = np.log(df['loan_amnt'] + 1)
    df['rate_times_amnt']     = df['loan_int_rate'] * df['loan_amnt']          # total interest burden
    df['credit_per_age']      = df['credit_score'] / (df['person_age'] + 1)   # credit maturity ratio

    # 2. Drop raw columns replaced by engineered ones
    df.drop(['person_income', 'loan_amnt'], axis=1, inplace=True)

    # 3. Encode binary categoricals
    df['person_gender'] = (df['person_gender'] == 'male').astype(int)
    df['previous_loan_defaults_on_file'] = (
        df['previous_loan_defaults_on_file'] == 'Yes'
    ).astype(int)

    # 4. Ordinal encode education (has natural order)
    edu_order = {
        'High School': 0,
        'Associate':   1,
        'Bachelor':    2,
        'Master':      3,
        'Doctorate':   4
    }
    df['person_education'] = df['person_education'].map(edu_order)

    # 5. One-hot encode nominal categoricals
    df = pd.get_dummies(
        df,
        columns=['person_home_ownership', 'loan_intent'],
        drop_first=True
    )

    # 6. Convert all bool columns from get_dummies to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


if __name__ == '__main__':
    df = load_and_preprocess('../data/raw/loan_data.csv')
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    df.to_csv('../data/processed/train_processed.csv', index=False)
    print("\nSaved to data/processed/train_processed.csv")