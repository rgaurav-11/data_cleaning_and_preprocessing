Titanic Dataset Preprocessing

This project cleans, preprocesses, and handles outliers from the Titanic dataset (`titanic.csv`). The end result is the cleaned dataset saved as `titanic_cleaned_final.csv`.

Objective

Preparing the Titanic dataset for machine learning by:

- Dealing with missing values
- Categorical encoding of variables
- Scaling numerical features
- Outlier detection and elimination
- Saving the cleaned data for future use
Files

- `titanic.csv`: Original Titanic dataset (must be in the working directory).
- `titanic_cleaned_final.csv`: Final cleaned dataset output.
- `titanic_preprocessing.py`: Python script used for cleaning the data (your script so far).
- `README.md`: This document.

Steps Performed

1. Load Dataset

python
df = pd.read_csv("titanic.csv")
The Titanic dataset is loaded using pandas.

2. Handle Missing Values
Age: Missing values filled with the median.

Embarked: Missing values filled with the most frequent value (mode).

Cabin: Entirely dropped due to excessive missing data.

Name and Ticket: Dropped as they're not useful for modeling.

3. Encode Categorical Variables
Sex: Label encoded → 'male' = 0, 'female' = 1.

Embarked: One-hot encoded (with first category dropped to avoid dummy trap).

4. Scale Numerical Features
Age and Fare are standardized to have a mean of 0 and standard deviation of 1.

5. Visualize Outliers
Boxplots are used to visualize outliers in the Age and Fare columns using seaborn.

6. Remove Outliers
Z-score is calculated for Age and Fare, and rows with a Z-score above 3 (absolute value) are removed.

7. Save Final Dataset
The cleaned dataset is exported for use in further analysis or machine learning modeling.

Output
A clean dataset saved as titanic_cleaned_final.csv
Boxplots showing distributions of Age and Fare before outlier removal
Terminal output of dataset info and shape at various stages

This cleaned dataset is now ready.
