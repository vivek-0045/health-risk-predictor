import sqlite3

conn = sqlite3.connect("../database/patients.db")
cursor = conn.cursor()

patients = [

    ("P001", 45, "Male", 92, 140, 90, 1, 180),
    ("P002", 38, "Female", 80, 130, 85, 0, 150),
    ("P003", 52, "Male", 98, 150, 95, 1, 210),
    ("P004", 29, "Female", 72, 120, 80, 0, 140),
    ("P005", 60, "Male", 105, 160, 100, 1, 230),
    ("P006", 41, "Female", 88, 135, 88, 0, 170),
    ("P007", 55, "Male", 96, 148, 92, 1, 200),
    ("P008", 33, "Female", 76, 125, 82, 0, 145),
    ("P009", 47, "Male", 90, 142, 90, 1, 185),
    ("P010", 50, "Female", 85, 138, 87, 0, 160),
    ("P011", 62, "Male", 108, 165, 102, 1, 240),
    ("P012", 36, "Female", 78, 128, 84, 0, 150),
    ("P013", 44, "Male", 94, 145, 91, 1, 195),
    ("P014", 28, "Female", 70, 118, 78, 0, 135),
    ("P015", 57, "Male", 102, 155, 97, 1, 220),
    ("P016", 39, "Female", 82, 132, 86, 0, 155),
    ("P017", 49, "Male", 95, 147, 93, 1, 205),
    ("P018", 31, "Female", 74, 122, 80, 0, 142),
    ("P019", 53, "Male", 99, 152, 96, 1, 215),
    ("P020", 42, "Female", 86, 136, 88, 0, 165),
    ("P021", 58, "Male", 104, 158, 99, 1, 225)

]

cursor.executemany("""
INSERT INTO patients
(patient_id, age, gender, heart_rate, systolic_bp, diastolic_bp, glucose_level, cholesterol)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", patients)

conn.commit()
conn.close()

print("21 patient records inserted successfully.")
