import mlflow
import mlflow.sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main(data_path):
    # Setup MLflow URI dan Auth dari environment variable (tidak interaktif)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment("student-passed-classifier")

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['math score', 'reading score', 'writing score', 'passed'])
    y = df['passed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Run pertama: baseline model
    with mlflow.start_run(run_name="baseline_rf_model"):
        baseline_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        baseline_model.fit(X_train, y_train)
        y_pred = baseline_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(baseline_model, artifact_path="rf_model_passed_classifier")

    # Tuning beberapa parameter
    best_model = None
    best_acc = 0

    for n in [10, 50, 100]:
        with mlflow.start_run(run_name=f"tuning_n_{n}"):
            tuned_model = RandomForestClassifier(n_estimators=n, random_state=42)
            tuned_model.fit(X_train, y_train)
            y_pred = tuned_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_param("n_estimators", n)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_0", report["0"]["precision"])
            mlflow.log_metric("recall_1", report["1"]["recall"])

            mlflow.sklearn.log_model(tuned_model, "model_rf")

            if acc > best_acc:
                best_model = tuned_model
                best_acc = acc

    # Simpan model terbaik ke file
    if best_model:
        joblib.dump(best_model, 'model.pkl')
        print("Model terbaik disimpan ke model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to cleaned student performance data")
    args = parser.parse_args()

    main(data_path=args.data_path)
