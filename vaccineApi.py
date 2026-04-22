from flask import Flask, request, jsonify
import pandas as pd
import joblib
import functools
import os

app = Flask(__name__)

# ==========================================
# 1. SETTINGS & SECURITY
# ==========================================
API_TOKEN = "your-super-secret-vaccine-token-123"

# Update these paths to your actual folders
BASE_PATH = r"C:\Users\user1\.venv\api\csv"
MODEL_PATH = r"C:\Users\user1\.venv\api\models"

VACCINE_MAP = {
    "1": {"name": "Human Papillomavirus (HPV) Vaccine", "csv": "HPV.csv", "model": "hpv_model.pkl"},
    "2": {"name": "Influenza Vaccine", "csv": "Influenza.csv", "model": "influenza_model.pkl"},
    "3": {"name": "Pneumococal Vaccine", "csv": "Pneumococal.csv", "model": "pneumococal_model.pkl"},
    "4": {"name": "Pneumococal Conjugate Vaccine (PCV)", "csv": "PCV.csv", "model": "pcv_model.pkl"},
    "5": {"name": "Inactivated Polio Vaccine (IPV)", "csv": "IPV.csv", "model": "ipv_model.pkl"},
    "6": {"name": "Measles, Mumps and Rubella Vaccine (MMR)", "csv": "MMR.csv", "model": "mmr_model.pkl"},
    "7": {"name": "Pentavalent Vaccine (DPT-Hep B-HiB)", "csv": "Pentavalent.csv", "model": "penta_model.pkl"},
    "8": {"name": "Oral Polio Vaccine (OPV)", "csv": "OPV.csv", "model": "opv_model.pkl"},
    "9": {"name": "Bacille Calmette-Guerin Vaccine (BCG)", "csv": "BCG.csv", "model": "bcg_model.pkl"},
    "10": {"name": "Hepatitis B Vaccine", "csv": "Hepatitis_B.csv", "model": "hepatitis_b_model.pkl"},
    "11": {"name": "Tetanus-Diphtheria Vaccine", "csv": "Tetanus_Diphtheria.csv", "model": "tetanus_diphtheria_model.pkl"},
    "12": {"name": "Vitamin K", "csv": "Vitamin_K.csv", "model": "vitamin_k_model.pkl"},
    "13": {"name": "Measles-Rubella Vaccine", "csv": "Measles_Rubella.csv", "model": "measles_rubella_model.pkl"}
}

def token_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        token = auth_header.split(" ")[1]
        if token != API_TOKEN:
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ==========================================
# 2. THE PREDICT ENDPOINT
# ==========================================
@app.route('/predict', methods=['POST'])
@token_required
def predict():
    content = request.get_json(silent=True)
    if not content or "data" not in content:
        return jsonify({"status": "error", "message": "Invalid JSON format"}), 422

    final_results = {}

    try:
        # Loop through the list of vaccine objects in the payload
        for vaccine_entry in content["data"]:
            for v_id, records in vaccine_entry.items():
                
                if v_id not in VACCINE_MAP:
                    continue # Skip vaccines not in our map
                
                v_info = VACCINE_MAP[v_id]
                current_csv = os.path.join(BASE_PATH, v_info["csv"])
                current_model = os.path.join(MODEL_PATH, v_info["model"])

                # --- STEP A: READ CSV SAFELY ---
                # We use skiprows=1 to avoid trying to parse the header 'Date' as a date
                df_existing = pd.read_csv(current_csv, header=None, names=['Date', 'Count'], skiprows=1)
                df_existing['Date'] = pd.to_datetime(df_existing['Date'])
                
                # Identify the current latest date in the file
                latest_date_in_file = df_existing['Date'].max()

                # --- STEP B: UPDATE OR APPEND ---
                for rec in records:
                    new_date = pd.to_datetime(rec['date'])
                    new_count = int(rec['count'])

                    if new_date == latest_date_in_file:
                        # Update the latest month's count
                        df_existing.loc[df_existing['Date'] == new_date, 'Count'] = new_count
                    elif new_date > latest_date_in_file:
                        # Append new future month
                        new_row = pd.DataFrame({'Date': [new_date], 'Count': [new_count]})
                        df_existing = pd.concat([df_existing, new_row], ignore_index=True)
                        latest_date_in_file = new_date # Update tracker for next record in payload
                    else:
                        # Ignore historical data changes
                        continue 

                # --- STEP C: SAVE UPDATED CSV ---
                df_existing.sort_values('Date', inplace=True)
                df_existing.to_csv(current_csv, header=True, index=False)

                # --- STEP D: FORECASTING ---
                df_model = df_existing.set_index('Date').asfreq('MS').fillna(0)
                model_fit = joblib.load(current_model)
                updated_results = model_fit.apply(df_model['Count'])
                
                # Get the number of steps from payload, default to 1
                num_steps = content.get('steps', 1)
                forecast_obj = updated_results.get_forecast(steps=num_steps)
                
                # Extract mean and confidence intervals
                mean_series = forecast_obj.predicted_mean
                conf_df = forecast_obj.conf_int()
                
                # Loop through all forecasted months and create a list
                forecast_list = []
                for i in range(len(mean_series)):
                    forecast_list.append({
                        "month": mean_series.index[i].strftime('%B %Y'),
                        "prediction": round(float(mean_series.iloc[i]), 2),
                        "lower_limit": round(float(conf_df.iloc[i, 0]), 2),
                        "upper_limit": round(float(conf_df.iloc[i, 1]), 2)
                    })
                
                # Store the full list for this specific vaccine
                final_results[v_info["name"]] = {
                    "total_steps_requested": num_steps,
                    "forecast_data": forecast_list
                }


        return jsonify({
            "status": "success",
            "results": final_results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    app.run(debug=False, port=5000)
