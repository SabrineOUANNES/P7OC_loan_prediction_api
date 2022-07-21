# Dependencies
from flask import Flask, request, jsonify
from functions import *
import json
import pandas as pd
import shap

# Your API definition
app = Flask(__name__)
# load model
model = load_model()
# Download datasets
data = pd.read_csv("datasets/train_data.csv", index_col='SK_ID_CURR')


@app.route('/api/predict', methods=['POST'])
def predict_customer_id():
    json_ = json.loads(request.data)
    id_client = json_["SK_ID_CURR"]
    check = check_id(id_client)
    if check:
        explanation = lime_explanation(id_client)
        client = data[data.index == id_client].drop(['TARGET'], axis=1)
        client_preproc = preprocessing(client)
        y_pred = model.predict(client_preproc)
        y_proba = model.predict_proba(client_preproc)
        y_proba = y_proba.tolist()
        res = {"statut_code": str(1),
               "prediction": str(y_pred[0]),
               "proba_yes": str(y_proba[0][0]),
               "proba_no": str(y_proba[0][1]),
               "lime_explanation": explanation.as_list()
               }
        return jsonify(res)
    else:
        res = {"statut_code": str(0),
               "erreur": "This ID doesn't exist"}
        return jsonify(res)


@app.route('/api/get/<id>', methods=['GET'])
def get_infos(id):
    check = check_id(int(id))
    if check:
        client = data[data.index == int(id)]
        client = client.values
        columns_df = list(data.columns)
        data_sample = data.values
        res = {"Features": columns_df,
               "customer_data": client.tolist(),
               "other_customers": data_sample.tolist()}

        return jsonify(res)
    else:
        res = {"statut_code": str(0),
               "erreur": "This ID doesn't exist"}
        return jsonify(res)


@app.route('/api/get/importance', methods=['GET'])
def feat_importance():
    feat_imp_columns = list(data.drop(columns='TARGET', axis=1).columns)
    feature_imp = model.feature_importances_
    feature_imp = list(map(str, feature_imp))
    res = {'features': feat_imp_columns,
           'Values': feature_imp}

    return jsonify(res)


@app.route('/api/shap/<id>', methods=['GET'])
def get_shap_info(id):
    check = check_id(int(id))
    if check:
        shap_data = data.drop(columns='TARGET', axis=1)
        shap_explainer = shap.TreeExplainer(model, shap_data.values)
        shap_values = shap_explainer.shap_values(shap_data.values)
        expec_value = str(shap_explainer.expected_value)
        shap_arr = list(shap_values[int(id)])
        client = shap_data[shap_data.index == int(id)].values.tolist()[0]
        feature_names = list(shap_data.columns)
        res = {
            'expec_value': expec_value,
            'shap_arr': shap_arr,
            'client': client,
            'feature_names': feature_names
        }
        return jsonify(res)
    else:
        res = {"statut_code": str(0),
               "erreur": "This ID doesn't exist"}
        return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)
