import pandas as pd
import numpy as np
import joblib
import json

# ── Load everything once at startup ──────────────────────
model_full     = joblib.load('model_full.pkl')
model_clinical = joblib.load('model_clinical.pkl')

with open('full_cols.json') as f:
    full_cols = json.load(f)

with open('clinical_cols.json') as f:
    clinical_cols = json.load(f)

with open('gene_cols.json') as f:
    gene_cols = json.load(f)


def predict_recurrence(clinical_data: dict, gene_data: dict = None):
    """
    clinical_data : dict of clinical feature values (required)
    gene_data     : dict of gene mutation values 0/1 (optional)

    Returns: probability, risk level, model used
    """

    if gene_data is not None:
        # ── Full prediction (clinical + genomic) ──────
        full_data = {**clinical_data, **gene_data}
        full_data['mutation_burden'] = sum(gene_data.values())

        df_input = pd.DataFrame([full_data])
        df_input = df_input.reindex(columns=full_cols, fill_value=0)

        prob = float(model_full.predict_proba(df_input)[:, 1][0])
        model_used = "Full model (clinical + genomic)"

    else:
        # ── Clinical only prediction ──────────────────
        df_input = pd.DataFrame([clinical_data])
        df_input = df_input.reindex(columns=clinical_cols, fill_value=0)

        prob = float(model_clinical.predict_proba(df_input)[:, 1][0])
        model_used = "Clinical model (no genomic data)"

    # Risk level
    if prob >= 0.7:
        risk = 'High'
    elif prob >= 0.4:
        risk = 'Medium'
    else:
        risk = 'Low'

    return {
        'recurrence_probability': round(prob * 100, 1),
        'risk_level': risk,
        'model_used': model_used
    }


# ── Test 1: Clinical only ─────────────────────────────────
result = predict_recurrence(
    clinical_data={
        'T_STAGE': 6.0,
        'N_STAGE': 3.0,
        'M_STAGE': 0.0,
        'HER2_STATUS_PRIMARY': 1.0,
        'MENOPAUSAL_STATUS_AT_DIAGNOSIS': 1.0,
        'STAGE_AT_DIAGNOSIS': 2.0,
        'OVERALL_TUMOR_GRADE': 2.0,
        'PRIMARY_NUCLEAR_GRADE': 2.0,
        'ER_PCT_PRIMARY': 90.0,
        'PR_PCT_PRIMARY': 80.0,
        'OVERALL_HER2_STATUS': 0.0,
        'INVASIVE_CARCINOMA_DX_AGE': 52,
        'OVERALL_RECEPTOR_STATUS_PATIENT_HR+/HER2-': 1,
        'RECEPTOR_STATUS_PRIMARY_HR+/HER2-': 1,
        'TUMOR_SAMPLE_HISTOLOGY_Breast Invasive Ductal Carcinoma': 1,
        'LATERALITY_Left': 1,
    },
    gene_data=None
)
print("Test 1 (clinical only):", result)


# ── Test 2: Full (clinical + genomic) ────────────────────
result_full = predict_recurrence(
    clinical_data={
        'T_STAGE': 6.0,
        'N_STAGE': 3.0,
        'M_STAGE': 0.0,
        'HER2_STATUS_PRIMARY': 1.0,
        'MENOPAUSAL_STATUS_AT_DIAGNOSIS': 1.0,
        'STAGE_AT_DIAGNOSIS': 2.0,
        'OVERALL_TUMOR_GRADE': 2.0,
        'PRIMARY_NUCLEAR_GRADE': 2.0,
        'ER_PCT_PRIMARY': 90.0,
        'PR_PCT_PRIMARY': 80.0,
        'OVERALL_HER2_STATUS': 0.0,
        'INVASIVE_CARCINOMA_DX_AGE': 52,
        'OVERALL_RECEPTOR_STATUS_PATIENT_HR+/HER2-': 1,
        'RECEPTOR_STATUS_PRIMARY_HR+/HER2-': 1,
        'TUMOR_SAMPLE_HISTOLOGY_Breast Invasive Ductal Carcinoma': 1,
        'LATERALITY_Left': 1,
    },
    gene_data={
        'PIK3CA': 1,
        'TP53': 0,
        'CDH1': 0,
        # all other genes default to 0 automatically
    }
)
print("Test 2 (full model):   ", result_full)