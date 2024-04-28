import numpy as np
import pandas as pd
import scipy.io as sc
from datetime import datetime, timedelta

# Provide the path to your .mat file
mat_file_path = 'Data/wiki.mat'

mat_contents = sc.loadmat(mat_file_path)

# Access the "wiki" variable
wiki_variable = mat_contents['wiki']

names = wiki_variable['name']

names_reshaped = names[0, 0].reshape(-1, 1)

# Convert the reshaped array of strings to a Pandas DataFrame
df_names = pd.DataFrame(names_reshaped, columns=['Name'])
# Convert the "Name" column to strings
df_names['Name'] = df_names['Name'].astype(str)
# Remove brackets from the names in the "Name" column
df_names['Name'] = df_names['Name'].str.replace('[', "").str.replace(']', "")
df_names['Name'] = df_names['Name'].str.replace("'", "")

gender = wiki_variable['gender']
gender_reshaped = gender[0, 0].reshape(-1, 1)

# Convert the "gender" array to a Pandas DataFrame
df_gender = pd.DataFrame(gender_reshaped, columns=['Gender'])

# Align the indices of df_names and df_gender
df_gender.index = df_names.index

# Concatenate df_names and df_gender along the columns axis
df_combined = pd.concat([df_names, df_gender], axis=1)
df_combined['Gender'] = df_combined['Gender'].replace({1: 'Male', 0: 'Female'})

birth_year = wiki_variable['dob']

# Reshape the birth_year array to 1D
birth_year_reshaped = birth_year[0, 0].reshape(-1, 1)

# Convert MATLAB serial date numbers to Python datetime objects
python_datetimes = np.array([datetime.fromordinal(int(dob)) + timedelta(days=float(dob % 1)) - timedelta(days=366) for dob in birth_year_reshaped.flatten()])

# Add the Python datetime objects to the DataFrame
df_combined['Birth Year'] = python_datetimes
df_combined['Birth Year'] = df_combined['Birth Year'].apply(lambda x: x.year)  # Extract the year directly
print(df_combined)

year_taken = wiki_variable['photo_taken']
year_taken_reshaped = year_taken[0, 0].reshape(-1, 1)
df_combined['Year taken'] = year_taken_reshaped
print(df_combined)
df_combined['At time years old'] = df_combined['Year taken'] - df_combined['Birth Year']
print(df_combined)