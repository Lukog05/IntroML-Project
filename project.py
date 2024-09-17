import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# Replace 'file_path.csv' with the actual path to your CSV file
file_path = 'heart_failure_clinical_records_dataset.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)



# Display the first few rows of the DataFrame
print(df.head())

# Some manipulation
# Assuming your DataFrame is named 'df'
"""
df["Parent Education Level"] = df["Parent Education Level"].fillna(np.nan)
df['Attendance Rate'] = df['Attendance Rate'].where(df['Attendance Rate'] <= 100.0, np.nan)
df['Previous Grades'] = df['Previous Grades'].where(df['Previous Grades'] <= 100.0, np.nan)
df['Attendance Rate'] = df['Attendance Rate'].where(df['Attendance Rate'] >= 0.0, np.nan)
df['Previous Grades'] = df['Previous Grades'].where(df['Previous Grades'] >= 0.0, np.nan)
df['Study Hours per Week'] = df['Study Hours per Week'].where(df['Study Hours per Week'] >= 0, np.nan)
df = df.dropna()
# Transforming nominal strings to numbers
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Mapping education rank to numbers
df['Parent Education Level'] = df['Parent Education Level'].map(education_rank)
df = df.drop(columns="Student ID")
print(df['Study Hours per Week'].min())
#print(df)
"""
corr = df.corr()
sb.heatmap(corr,annot=True)


data = df.to_numpy()
print(data.shape)
fig, ax = plt.subplots(2,2)

bins, value_counts = np.unique(data[:,-1], return_counts=True)
ax[0][0].bar(bins,value_counts)
ax[0][0].set_xlabel(r'Died or not during follow-up period')
ax[0][0].set_ylabel(r'Count')


color = ['red' if d == 0 else 'blue' for d in data[:,5]]
platelets = data[:,6] / data[:,6].max()
ax[0][1].scatter(data[:,0],data[:,4], c=color,  s =platelets*100, alpha=0.5)
ax[0][1].set_xlabel(r'Age')
ax[0][1].set_ylabel(r'Ejection Fraction')
ax[0][1].set_title('Size = platelets, color = High blood pressure')


color2 = ['red' if d == 0 else 'blue' for d in data[:,9]]

ax[1][0].scatter(data[:,0], data[:,-3], c=color2, s=platelets*100)
ax[1][0].set_xlabel(r'Age')
ax[1][0].set_ylabel(r'Smokes')
ax[1][0].set_title('Size = platelets, color = Sex')



"""
color = ['red' if d == 0 else 'blue' for d in data[:,-1]]
color2 = ['red' if d == 0 else 'blue' for d in data[:,3]]

fig, ax = plt.subplots(2,2)

ax[0][0].scatter(data[:,0], data[:,1], c=color, s=data[:,2], alpha=0.5)

ax[0][0].set_xlabel(r'Study hours per week', fontsize=15)
ax[0][0].set_ylabel(r'Attendance rate', fontsize=15)
ax[0][0].set_title('Volume and percent change')

ax[0][1].scatter(data[:,0], data[:,2], c=color2, s=1, alpha=0.5)

ax[0][1].set_xlabel(r'Study hours per week', fontsize=15)
ax[0][1].set_ylabel(r'Previous Grade', fontsize=15)
ax[0][1].set_title('Volume and percent change')

bins, value_counts = np.unique(data[:,-2], return_counts=True)
#value_counts, bins = np.histogram(data[:,-2],bins=np.unique(data[:,-2]))
print(data[:,-2].dtype)
#print(bins)
#print(value_counts)
#value_counts = value_counts/np.shape(data)[0] * 100
#print(value_counts)

ax[1][0].bar( bins, value_counts)
ax[1][0].set_xlabel(r'Parent Education Level', fontsize=10)
ax[1][0].set_ylabel(r'Count', fontsize=10)
ax[1][0].set_title('Volume and percent change')
#ax.grid(True)

"""
fig.tight_layout()


plt.show()
