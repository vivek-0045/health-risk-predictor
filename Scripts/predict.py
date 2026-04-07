import sqlite3
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import smtplib
from email.message import EmailMessage

# -----------------------------
# 0. Load trained model
# -----------------------------
model = joblib.load("../model/health_risk_model.pkl")

# -----------------------------
# 1. Choose Prediction Mode
# -----------------------------
print("\nSelect Prediction Mode:")
print("1 → Fetch Patient from Database")
print("2 → Enter Features Manually")

choice = input("\nEnter choice (1/2): ").strip()

# -----------------------------
# 2A. OPTION 1 — Fetch from DB
# -----------------------------
if choice == "1":

    patient_id = input("\nEnter Patient ID: ").strip()

    conn = sqlite3.connect("../database/patients.db")

    query = """
    SELECT * FROM patients
    WHERE patient_id = ?
    """

    df = pd.read_sql_query(query, conn, params=(patient_id,))
    conn.close()

    if df.empty:
        print("❌ Patient ID not found.")
        sys.exit()

    print("\nPatient Record:\n")
    print(df)

# -----------------------------
# 2B. OPTION 2 — Manual Entry
# -----------------------------
elif choice == "2":

    print("\nEnter Patient Details:\n")

    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    systolic_bp = float(input("Systolic BP: "))
    diastolic_bp = float(input("Diastolic BP: "))
    cholesterol = float(input("Cholesterol: "))
    glucose = float(input("Glucose Level: "))
    heart_rate = float(input("Heart Rate: "))

    df = pd.DataFrame([{
        "patient_id": "Manual",
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "glucose_level": glucose,
        "heart_rate": heart_rate
    }])

else:
    print("❌ Invalid choice.")
    sys.exit()

# -----------------------------
# 3. Preprocess data
# -----------------------------
df_model = df.drop(columns=["patient_id"])

df_model["gender"] = df_model["gender"].map({
    "Male": 1,
    "Female": 0,
    "male": 1,
    "female": 0
})

df_model = df_model[model.feature_names_in_]

# -----------------------------
# 4. Prediction
# -----------------------------
prediction = model.predict(df_model)[0]
probability = model.predict_proba(df_model)[0][1]

print("\nPrediction Result:\n")
print(f"Risk Probability: {probability:.2f}")

# -----------------------------
# 4A. Risk Level Stratification
# -----------------------------
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

print(f"Risk Level: {risk_level}")

# -----------------------------
# 5. Risk Type Classification
# -----------------------------
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

if len(risk_type) == 0:
    risk_type.append("General Health Risk")

print("\nRisk Type(s):")
for r in risk_type:
    print("•", r)

# -----------------------------
# 6. SHAP Explainability
# -----------------------------
print("\nGenerating SHAP Explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_model)

if isinstance(shap_values, list):
    shap_val = shap_values[1][0]
    base_val = explainer.expected_value[1]
else:
    if len(shap_values.shape) == 2:
        shap_val = shap_values[0]
        base_val = explainer.expected_value
    elif len(shap_values.shape) == 3:
        shap_val = shap_values[0, :, 1]
        base_val = explainer.expected_value[1]
    else:
        raise ValueError("Unexpected SHAP values shape")

# -----------------------------
# 6A. Save Waterfall Plot
# -----------------------------
waterfall_path = "waterfall.png"

shap.plots.waterfall(
    shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=df_model.iloc[0],
        feature_names=df_model.columns
    ),
    show=False
)

plt.savefig(waterfall_path, bbox_inches="tight")
plt.close()

# -----------------------------
# 6B. Save Force Plot
# -----------------------------
force_path = "force.png"

shap.force_plot(
    base_val,
    shap_val,
    df_model.iloc[0],
    matplotlib=True,
    show=False
)

plt.savefig(force_path, bbox_inches="tight")
plt.close()

print("SHAP graphs saved.")

# -----------------------------
# 7. Ask Email Option
# -----------------------------
send_mail = input("\nDo you want to send report via Email? (y/n): ")

if send_mail.lower() != "y":
    print("Report not sent.")
    sys.exit()

receiver_email = input("Enter patient email: ")

# -----------------------------
# 8. Email Sending
# -----------------------------
print("\nSending Email Report...")

EMAIL_ADDRESS = "healthrisk.system@gmail.com"
EMAIL_PASSWORD = "hnhr mrks fhsl hsfy"

msg = EmailMessage()
msg["Subject"] = "Health Risk Prediction Report"
msg["From"] = EMAIL_ADDRESS
msg["To"] = receiver_email

msg.set_content(f"""
Health Risk Prediction Report

Risk Probability : {probability:.2f}
Risk Level       : {risk_level}

Risk Type(s):
{chr(10).join(risk_type)}

Generated by Explainable AI Health Risk Prediction System.
""")

# Attach SHAP graphs
for file in [waterfall_path, force_path]:
    with open(file, "rb") as f:
        data = f.read()
        name = f.name

    msg.add_attachment(
        data,
        maintype="image",
        subtype="png",
        filename=name
    )

# Send email
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("✅ Email sent successfully.")
