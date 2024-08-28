# from flask import Flask, jsonify, request, render_template
from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import streamlit as st
import os
import time

app = Flask(__name__)
print(pd.__version__)
current_folder = os.getcwd()
basepath = os.path.join(current_folder, "Models")

# load models, threshold, data and explainer
model_load = joblib.load(os.path.join(basepath, "model.pkl"))
best_thresh = joblib.load(os.path.join(basepath, "best_thresh_LightGBM_NS.pkl"))
X_test = pd.read_csv(os.path.join(basepath, "X_test_sample.csv"), index_col=0)
y_test = pd.read_csv(os.path.join(basepath, "y_test_sample.csv"), index_col=0)
shap_values = pd.read_csv(os.path.join(basepath, "shap_values_sample.csv"), index_col=0)
shap_values1 = pd.read_csv(os.path.join(basepath, "shap_values1_sample.csv"), index_col=0)
# Liste des clients à supprimer
clients_a_supprimer = [136718, 307488, 378985]

data = pd.DataFrame(y_test, index=y_test.index).reset_index()
# Supprimer les clients
data = data[~data['SK_ID_CURR'].isin(clients_a_supprimer)]

# Optionnel : sauvegarder les données filtrées
data.to_csv(os.path.join(basepath, "X_test_filtered.csv"), index=False)

columns = joblib.load('Models/columns.pkl')                                
# Compute feature importance
# compute mean of absolute values for shap values
vals = np.abs(shap_values1).mean(0)
# compute feature importance as a dataframe containing vals
feature_importance = pd.DataFrame(list(zip(columns, vals)),\
    columns=['col_name','feature_importance_vals'])
# Define top 10 features for customer details
top_10 = feature_importance.sort_values(by='feature_importance_vals', ascending=False)[0:10].col_name.tolist()
# Define top 20 features for comparison vs group
top_20 = feature_importance.sort_values(by='feature_importance_vals', ascending=False)[0:20].col_name.tolist()
feat_tot = feature_importance.feature_importance_vals.sum()
feat_top = feature_importance.loc[feature_importance['col_name'].isin(top_20)].feature_importance_vals.sum()

@app.route("/", methods=['GET']) #hello_world() sera appelée lorsque l'utilisateur accède à la racine de l'application (l'URL /)"
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict/<int:Client_Id>", methods=['GET'])
def predict(Client_Id: int):
    start_time = time.time()
    try:
        if 'SK_ID_CURR' not in data.columns:
            return jsonify({"error": "La colonne 'SK_ID_CURR' n'existe pas dans les données."}), 500
        
        print(f"Recherche de Client_Id: {Client_Id}")
        
        filtered_data = data.loc[data["SK_ID_CURR"] == int(Client_Id)]
        
        if filtered_data.empty:
            return jsonify({"error": f"Client_Id {Client_Id} non trouvé."}), 404
        
        data_idx = filtered_data.index[0]
        ID_to_predict = pd.DataFrame(X_test.iloc[data_idx, :]).T 
        
        prediction = sum((model_load.predict_proba(ID_to_predict)[:, 1] > best_thresh) * 1)
        
        decision = "granted" if prediction == 0 else "not granted"
        prob_predict = float(model_load.predict_proba(ID_to_predict)[:, 1])
        
        response = {
            "decision": decision,
            "prediction": int(prediction),
            "prob_predict": prob_predict,
            "ID_to_predict": ID_to_predict.to_json(orient='columns')
        }
        
        print(response)
        print("Temps pris: {:.2f} secondes".format(time.time() - start_time))
        return jsonify(response)
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({"error": str(e)}), 500

# provide data for shap features importance on selected customer's credit decision 
@app.route("/cust_vs_group/<int:Client_Id>", methods=['GET'])
def cust_vs_group(Client_Id: int):
    start_time = time.time()
    try:
        # Vérifiez si 'SK_ID_CURR' existe
        if 'SK_ID_CURR' not in data.columns:
            return jsonify({"error": "La colonne 'SK_ID_CURR' n'existe pas dans les données."}), 500
        
        print(f"Recherche de Client_Id: {Client_Id}")
        
        # Filtrer les données pour trouver le Client_Id
        filtered_data = data.loc[data["SK_ID_CURR"] == int(Client_Id)]
        
        # Vérifiez si des données existent pour le Client_Id
        if filtered_data.empty:
            return jsonify({"error": f"Client_Id {Client_Id} non trouvé."}), 404
        
        data_idx = filtered_data.index[0]
        ID_to_predict = pd.DataFrame(X_test.iloc[data_idx, :]).T
        
        # Réaliser la prédiction
        prediction = sum((model_load.predict_proba(ID_to_predict)[:, 1] > best_thresh) * 1)
        decision = "granted" if prediction == 0 else "not granted"
        
        print(f"Temps pris: {time.time() - start_time:.2f} secondes")
        
        # Retourner la réponse en JSON
        return jsonify({
            'decision': decision,
            'base_value': shap_values.base_values[data_idx],
            'shap_values1_idx': shap_values1[data_idx, :].tolist(),
            "ID_to_predict": ID_to_predict.to_json(orient='columns')
        })
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/load_top_10/", methods=['GET'])
def load_top_10():
    return json.dumps({"top_10" : top_10})

@app.route("/load_top_20/", methods=['GET'])
def load_top_20():
    return json.dumps({"top_20" : top_20, 'feat_tot': feat_tot, 'feat_top': feat_top})

@app.route("/load_best_thresh/", methods=['GET'])
def load_best_thresh():
    return {"best_thresh" : best_thresh} 

@app.route("/load_X_test/", methods=['GET'])
def load_X_test():
    return {"X_test" : pd.DataFrame(X_test).to_json(orient='columns')} 

@app.route("/load_data/", methods=['GET'])
def load_data():
    return {"data" : pd.DataFrame(data).to_json(orient='columns')} 

@app.route("/list_clients", methods=['GET'])
def list_clients():
    return jsonify(data['SK_ID_CURR'].unique().tolist())

if __name__ == "__main__":
    print("Starting server on port 8500")
    print("Running...")
    app.run(port=8500,debug=True , use_reloader=False)
    
    print("Stopped")
#