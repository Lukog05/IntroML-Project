import pandas as pd
import numpy as np

education_rank = {
    'None': 1,
    'High School': 2,
    'Associate': 3,
    'Bachelor': 4,
    'Master': 5,
    'Doctorate': 6
}


# Replace 'file_path.csv' with the actual path to your CSV file
file_path = 'student_performance_prediction.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
#print(df.head())

# Input Sanitization
for index, row in df.iterrows():
    if row["Study Hours per Week"] < 0:
        row["Study Hours per Week"] = "NaN"
    if row["Attendance Rate"] > 100:
        row["Attendance Rate"] = "NaN"
    if row["Previous Grades"] > 100:
        row["Previous Grades"] = "NaN"


df["Parent Education Level"].fillna("None",inplace=True)

df = df.dropna()

# Transforming nominal strings to numbers
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Mapping education rank to numbers
df['Parent Education Level'] = df['Parent Education Level'].map(education_rank)
df = df.drop(columns="Student ID")
print(df)

data = df.to_numpy()
print(data)