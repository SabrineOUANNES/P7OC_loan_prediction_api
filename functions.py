import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lime import lime_tabular

data = pd.read_csv("datasets/train_data.csv", index_col='SK_ID_CURR')


# datasets preprocessing
def preprocessing(X):
    scaler = StandardScaler()
    scaler.fit_transform(data.drop(columns=['TARGET']))
    X = scaler.transform(X)
    return X


#list id
def check_id(id_client):
    customers_id_list = list(data.index.sort_values())
    if id_client in customers_id_list :
        return True
    else :
        return False


# To download model
def load_model():
    filename = 'model/finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return model

model = load_model()


def lime_data():
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data = data.drop(columns=['TARGET'], axis=1).values,
        feature_names = list(data.drop(columns=['TARGET']).columns),
        class_names=[0, 1],
        mode='classification',
        verbose=True,
        random_state=1
    )
    return lime_explainer


def lime_explanation(idx):
    lime_explainer = lime_data()
    explanation = lime_explainer.explain_instance(
    data_row=data.drop(columns=['TARGET'], axis=1).values[idx],
    predict_fn=model.predict_proba,
    num_features=20)
    return explanation