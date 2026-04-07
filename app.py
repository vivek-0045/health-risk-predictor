import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, session
import sqlite3
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
import os

app = Flask(__name__)
app.secret_key = "secret123"

print("🚀 APP STARTING...")

# -----------------------------
# SAFE PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "health_risk_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "database", "patients.db")

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
model = None

def load_model():
    global model
    if model is None:
        print("📦 Loading model...")
        model = joblib.load(MODEL_PATH)
    return model


# -----------------------------
# HOME
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------
# PREDICT
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    model = load_model()

    mode = request.form.get("mode")

    if mode == "db":
        patient_id = request.form.get("patient_id")

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM patients WHERE patient_id = ?",
            conn,
            params=(patient_id,)
        )
        conn.close()

        if df.empty:
            return "❌ Patient ID not found"

    else:
        df = pd.DataFrame([{
            "patient_id": "Manual",
            "age": int(request.form['age']),
            "gender": request.form['gender'],
            "systolic_bp": float(request.form['systolic_bp']),
            "diastolic_bp": float(request.form['diastolic_bp']),
            "cholesterol": float(request.form['cholesterol']),
            "glucose_level": float(request.form['glucose_level']),
            "heart_rate": float(request.form['heart_rate'])
        }])

    df_model = df.drop(columns=["patient_id"])

    df_model["gender"] = df_model["gender"].map({
        "Male": 1, "Female": 0,
        "male": 1, "female": 0
    })

    df_model = df_model[model.feature_names_in_]

    probability = model.predict_proba(df_model)[0][1]

    # Risk level
    if probability <= 0.20:
        risk_level = "Very Low Risk"
    elif probability <= 0.35:
        risk_level = "Low Risk"
    elif probability <= 0.50:
        risk_level = "Moderate Risk"
    elif probability <= 0.65:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    # Risk types
    row = df.iloc[0]
    risk_type = []

    if row["cholesterol"] > 200:
        risk_type.append("Cardiovascular Risk")
    if row["systolic_bp"] > 140 or row["diastolic_bp"] > 90:
        risk_type.append("Hypertension Risk")
    if row["glucose_level"] > 125:
        risk_type.append("Diabetes Risk")
    if row["heart_rate"] > 100:
        risk_type.append("Arrhythmia Risk")

    if not risk_type:
        risk_type.append("General Health Risk")

    # -----------------------------
    # SHAP (SAFE TRY)
    # -----------------------------
    try:
        print("📊 Generating SHAP...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_model)

        if isinstance(shap_values, list):
            shap_val = shap_values[1][0]
            base_val = explainer.expected_value[1]
        else:
            shap_val = shap_values[0]
            base_val = explainer.expected_value

        os.makedirs("static", exist_ok=True)

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_val,
                base_values=base_val,
                data=df_model.iloc[0],
                feature_names=df_model.columns
            ),
            show=False
        )
        plt.savefig("static/waterfall.png", bbox_inches="tight")
        plt.close()

    except Exception as e:
        print("⚠️ SHAP ERROR:", e)

    session["probability"] = probability
    session["risk_level"] = risk_level
    session["risk_type"] = risk_type

    return render_template(
        "result.html",
        probability=round(probability, 2),
        risk_level=risk_level,
        risk_type=risk_type
    )


# -----------------------------
# EMAIL (ENV VARIABLES FIX)
# -----------------------------
@app.route('/send_mail', methods=['POST'])
def send_mail():

    data = request.get_json()
    receiver_email = data.get("email")

    EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return "Email config missing", 500

    msg = EmailMessage()
    msg["Subject"] = "Health Risk Report"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = receiver_email

    msg.set_content(f"""
Risk Probability : {session.get('probability')}
Risk Level       : {session.get('risk_level')}
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    return "Email Sent"


# -----------------------------
# NO app.run for Azure
# -----------------------------