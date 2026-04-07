import sqlite3

conn = sqlite3.connect("../database/patients.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    heart_rate INTEGER,
    systolic_bp INTEGER,
    diastolic_bp INTEGER,
    glucose_level INTEGER,
    cholesterol INTEGER 
)
""")

conn.commit()
conn.close()

print("Database created with updated schema successfully.")
