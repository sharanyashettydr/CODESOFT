import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


file_path = 'Churn_Modelling.csv'


if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the CSV is in the same directory as this script.")
    exit()

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully.")
    print("\nDataset head:")
    print(df.head())
    print("\nDataset info:")
    df.info()
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()




df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
print("\n✅ Dropped irrelevant identifier columns.")
print(f"Dataset shape after dropping columns: {df.shape}")

X = df.drop('Exited', axis=1)
y = df['Exited']

print("\n✅ Separated features (X) and target (y).")


categorical_cols = ['Geography', 'Gender']
print(f"\nCategorical columns to be encoded: {categorical_cols}")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)

print("\n✅ Categorical features one-hot encoded.")
print("Updated features shape:", X.shape)
print("Updated features columns:\n", X.columns.tolist())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\n✅ Data split into training and testing sets.")
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")


numerical_cols = X_train.select_dtypes(include=np.number).columns


scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])


X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("\n✅ Numerical features scaled (using scaler fitted on training data).")


models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Exited (0)', 'Exited (1)'])
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"✅ {name} trained and evaluated.")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


print("\n--- Prediction on New Data (Example) ---")

sample_data = {
    'CreditScore': [600], 'Geography': ['France'], 'Gender': ['Male'], 'Age': [40],
    'Tenure': [3], 'Balance': [60000.0], 'NumOfProducts': [2], 'HasCrCard': [1],
    'IsActiveMember': [1], 'EstimatedSalary': [50000.0]
}
sample_df = pd.DataFrame(sample_data)


categorical_cols_for_sample = ['Geography', 'Gender'] # Use the same list
sample_encoded = pd.get_dummies(sample_df, columns=categorical_cols_for_sample, drop_first=True, dtype=float)


missing_cols = set(X_train.columns) - set(sample_encoded.columns)
for col in missing_cols:
    sample_encoded[col] = 0

sample_encoded = sample_encoded[X_train.columns]


sample_encoded[numerical_cols] = scaler.transform(sample_encoded[numerical_cols])


best_model = models['Random Forest']
prediction_label = best_model.predict(sample_encoded)[0]
prediction_proba = best_model.predict_proba(sample_encoded)[0, 1]

print("\n--- Prediction for a Sample Customer ---")
print(f"Features of the sample customer:\n{sample_df.iloc[0].to_string()}")
print(f"\nThe model predicts the customer will {'exit (churn)' if prediction_label == 1 else 'not exit (stay)'}.")
print(f"Predicted churn probability: {prediction_proba:.4f}")
print("-" * 40)