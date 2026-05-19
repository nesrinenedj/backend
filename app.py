

import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load everything once at startup ──────────────────────
model_full          = joblib.load('model_knn_adasyn_full.pkl')
model_clinical      = joblib.load('model_knn_adasyn_clinical.pkl')
ordinal_encoder     = joblib.load('ordinal_encoder.pkl')
nominal_encoder     = joblib.load('nominal_encoder.pkl')
imputer_full        = joblib.load('imputer_full.pkl')
imputer_clinical    = joblib.load('imputer_clinical.pkl')

with open('full_cols.json') as f:
    full_cols = json.load(f)

with open('clinical_cols.json') as f:
    clinical_cols = json.load(f)

with open('gene_cols.json') as f:
    gene_cols = json.load(f)

# ── Column definitions ────────────────────────────────────
ordinal_col_names = [
    'T_STAGE', 'N_STAGE', 'M_STAGE', 'STAGE_AT_DIAGNOSIS',
    'OVERALL_TUMOR_GRADE', 'PRIMARY_NUCLEAR_GRADE',
    'MENOPAUSAL_STATUS_AT_DIAGNOSIS', 'HER2_STATUS_PRIMARY',
    'OVERALL_HER2_STATUS'
]

nominal_col_names = [
    'OVERALL_RECEPTOR_STATUS_PATIENT',
    'SEX',
    'LATERALITY',
    'RECEPTOR_STATUS_PRIMARY',
    'TUMOR_SAMPLE_HISTOLOGY'
]


def encode(data: dict) -> pd.DataFrame:
    """
    Takes raw clinical data (strings + numbers from frontend),
    applies ordinal encoding and nominal encoding.
    Returns a dataframe with encoded columns, NaN still present.
    """
    df = pd.DataFrame([data])

    # ── 1. Ordinal encoding ───────────────────────────────
    cols_to_encode = [c for c in ordinal_col_names if c in df.columns]
    if cols_to_encode:
        df[cols_to_encode] = ordinal_encoder.transform(df[cols_to_encode])

    # ── 2. Nominal encoding ───────────────────────────────
    cols_to_onehot = [c for c in nominal_col_names if c in df.columns]
    if cols_to_onehot:
        nominal_input = df[cols_to_onehot].fillna('__MISSING__')
        encoded = pd.DataFrame(
            nominal_encoder.transform(nominal_input),
            columns=nominal_encoder.get_feature_names_out(cols_to_onehot),
            index=df.index
        )

        missing_cols = [c for c in encoded.columns if '__MISSING__' in c]
        encoded = encoded.drop(columns=missing_cols)

        df = df.drop(columns=cols_to_onehot)
        df = pd.concat([df, encoded], axis=1)

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()

        if not body:
            return jsonify({'error': 'Request body is empty'}), 400

        clinical_data = body.get('clinical')
        gene_data     = body.get('genes', None)

        if not clinical_data:
            return jsonify({'error': 'Missing required field: clinical'}), 400

        # ── Validate required clinical fields ─────────────
        required_fields = [
            'T_STAGE', 'N_STAGE', 'M_STAGE', 'STAGE_AT_DIAGNOSIS',
            'INVASIVE_CARCINOMA_DX_AGE', 'SEX',
            'MENOPAUSAL_STATUS_AT_DIAGNOSIS', 'LATERALITY',
            'OVERALL_TUMOR_GRADE', 'PRIMARY_NUCLEAR_GRADE',
            'HER2_STATUS_PRIMARY', 'OVERALL_HER2_STATUS',
            'OVERALL_RECEPTOR_STATUS_PATIENT',
            'RECEPTOR_STATUS_PRIMARY', 'TUMOR_SAMPLE_HISTOLOGY'
        ]

        for field in required_fields:
            if field not in clinical_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        if gene_data is not None:
            # ── Full model (clinical + genomic) ───────────
            gene_dict = {g: 0 for g in gene_cols if g != 'mutation_burden'}
            gene_dict.update({k: v for k, v in gene_data.items()
                              if k in gene_dict})
            gene_dict['mutation_burden'] = sum(gene_dict.values())

            full_data  = {**clinical_data, **gene_dict}
            df_encoded = encode(full_data)
            df_encoded = df_encoded.reindex(columns=full_cols, fill_value=np.nan)


            df_ready = pd.DataFrame(
                imputer_full.transform(df_encoded),
                columns=full_cols
            )

            prob = float(model_full.predict_proba(df_ready)[:, 1][0])
            model_used = "Full model (clinical + genomic)"

        else:
            # ── Clinical only model ───────────────────────
            df_encoded = encode(clinical_data)
            df_encoded = df_encoded.reindex(columns=clinical_cols, fill_value=np.nan)


            df_ready = pd.DataFrame(
                imputer_clinical.transform(df_encoded),
                columns=clinical_cols
            )

            prob = float(model_clinical.predict_proba(df_ready)[:, 1][0])
            model_used = "Clinical model (no genomic data)"

        # ── Binary prediction at 0.5 threshold ───────────
        prediction = 'Recurrence likely' if prob >= 0.5 else 'Recurrence unlikely'

        return jsonify({
            'recurrence_probability': round(prob * 100, 1),
            'prediction': prediction,
            'model_used': model_used
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)