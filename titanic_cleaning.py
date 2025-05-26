import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("titanic.csv")

# Step 2: Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Step 3: Encode categorical variables
# Label encode 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked' and convert bools to int
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True).astype(int)
df = pd.concat([df.drop('Embarked', axis=1), embarked_dummies], axis=1)

# Print summary
print("Cleaned and encoded data:")
print(df.head())

print("Missing values:")
print(df.isnull().sum())

print(" Column types:")
print(df.dtypes)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Choose numeric columns to scale
cols_to_scale = ['Age', 'Fare']

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("Scaled numerical features:")
print(df[cols_to_scale].head())
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for Age
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

# Boxplot for Fare
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()
from scipy import stats
import numpy as np

# Remove rows where Age or Fare is an outlier
z_scores = np.abs(stats.zscore(df[['Age', 'Fare']]))
df = df[(z_scores < 3).all(axis=1)]

print("Dataset shape after removing outliers:", df.shape)
df.to_csv("titanic_cleaned_final.csv", index=False)
print("Final cleaned dataset saved as titanic_cleaned_final.csv")
