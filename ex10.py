import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
 
RANDOM_STATE = 42
DATA_FILE = "customer_churn.csv"
MODEL_FILE = "churn_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"
 
 
# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------
def load_data(n_samples=2000):
    """Load real CSV if available, otherwise generate a synthetic dataset
    that mimics a typical telecom churn dataset (similar structure to the
    popular 'Telco Customer Churn' dataset)."""
    if os.path.exists(DATA_FILE):
        print(f"Loading dataset from '{DATA_FILE}'...")
        df = pd.read_csv(DATA_FILE)
        return df
 
    print("No CSV found — generating a synthetic churn dataset instead.")
    rng = np.random.default_rng(RANDOM_STATE)
 
    gender = rng.choice(["Male", "Female"], n_samples)
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.85, 0.15])
    tenure = rng.integers(0, 73, n_samples)                     # months
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
    )
    tech_support = rng.choice(["Yes", "No"], n_samples, p=[0.4, 0.6])
    monthly_charges = rng.normal(65, 25, n_samples).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n_samples)).clip(0)
 
    # Build churn probability from a realistic combination of factors
    churn_prob = (
        0.45 * (contract == "Month-to-month")
        + 0.20 * (internet_service == "Fiber optic")
        + 0.15 * (tech_support == "No")
        + 0.15 * (payment_method == "Electronic check")
        + 0.20 * (tenure < 12)
        + 0.10 * (monthly_charges > 80)
        - 0.20 * (tenure > 48)
    )
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
    churn = rng.binomial(1, churn_prob.clip(0.03, 0.9))
 
    df = pd.DataFrame({
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Tenure": tenure,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "TechSupport": tech_support,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges.round(2),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df
 
 
# ---------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
def explore_data(df):
    print("\n--- Dataset Overview ---")
    print(df.shape)
    print(df.head())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Churn Distribution ---")
    print(df["Churn"].value_counts(normalize=True).round(3))
 
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Churn", hue="Churn", data=df, palette="Set2", legend=False)
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig("churn_distribution.png")
    plt.close()
    print("Saved churn_distribution.png")
 
 
# ---------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------
def preprocess_data(df):
    df = df.copy()
 
    # Drop obvious ID columns if present
    for col in ["CustomerID", "customerID", "ID"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
 
    # Ensure TotalCharges is numeric (real datasets often store it as text)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.fillna({"TotalCharges": df["TotalCharges"].median()}, inplace=True)
 
    target_col = "Churn"
    # Map Yes/No (or True/False, 1/0) to 0/1 regardless of the column's exact
    # dtype (pandas versions differ on whether text columns show as
    # 'object' or 'str', so we check the actual values instead of dtype).
    unique_vals = set(df[target_col].dropna().unique().tolist())
    if unique_vals <= {"Yes", "No"}:
        y = df[target_col].map({"Yes": 1, "No": 0})
    elif unique_vals <= {True, False}:
        y = df[target_col].astype(int)
    else:
        y = df[target_col]
    y = y.astype(int)
    X = df.drop(columns=target_col)
 
    encoders = {}
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
 
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
 
    return X, y, scaler, encoders, list(categorical_cols), list(numeric_cols)
 
 
# ---------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models
 
 
# ---------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------
def evaluate_models(models, X_test, y_test):
    results = {}
    plt.figure(figsize=(6, 5))
 
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
 
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
 
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec,
                          "f1": f1, "roc_auc": auc}
 
        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")
        print(f"ROC-AUC  : {auc:.3f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
 
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
 
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
    print("Saved roc_curve.png")
 
    # Save a comparison table (useful to paste straight into a report)
    comparison_df = pd.DataFrame(results).T.round(3)
    comparison_df.to_csv("model_comparison.csv")
    print("Saved model_comparison.csv")
    print("\n--- Model Comparison ---")
    print(comparison_df)
 
    comparison_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(8, 5)
    )
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    print("Saved model_comparison.png")
 
    return results
 
 
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
 
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("Saved feature_importance.png")
 
 
# ---------------------------------------------------------------------
# 6. PREDICTION ON NEW DATA
# ---------------------------------------------------------------------
def predict_new_customer(model, scaler, encoders, categorical_cols,
                          numeric_cols, feature_order, customer_dict):
    """
    customer_dict: a dict with the same feature names used in training,
    e.g. {"Gender": "Female", "SeniorCitizen": 0, "Tenure": 5,
          "Contract": "Month-to-month", "InternetService": "Fiber optic",
          "PaymentMethod": "Electronic check", "TechSupport": "No",
          "MonthlyCharges": 85.5, "TotalCharges": 427.5}
    """
    row = pd.DataFrame([customer_dict])[feature_order]
 
    for col in categorical_cols:
        le = encoders[col]
        value = row.at[0, col]
        if value not in le.classes_:
            raise ValueError(
                f"Unknown value '{value}' for '{col}'. "
                f"Expected one of: {list(le.classes_)}"
            )
        row[col] = le.transform(row[col])
 
    row[numeric_cols] = scaler.transform(row[numeric_cols])
 
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    return ("Yes" if pred == 1 else "No"), round(prob * 100, 2)
 
 
# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_data()
    explore_data(df)
 
    X, y, scaler, encoders, categorical_cols, numeric_cols = preprocess_data(df)
    feature_order = list(X.columns)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
 
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
 
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]
    print(f"\nBest model based on ROC-AUC: {best_name}")
 
    plot_feature_importance(models["Random Forest"], feature_order)
 
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoders, ENCODERS_FILE)
    print(f"\nSaved model -> {MODEL_FILE}, scaler -> {SCALER_FILE}, encoders -> {ENCODERS_FILE}")
 
    # Example single prediction
    sample_customer = {
        "Gender": "Female",
        "SeniorCitizen": 0,
        "Tenure": 5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5,
    }
    label, prob = predict_new_customer(
        best_model, scaler, encoders, categorical_cols, numeric_cols,
        feature_order, sample_customer
    )
    print(f"\nSample prediction -> Churn: {label} (probability: {prob}%)")
 
 
if __name__ == "__main__":
    main()
 
