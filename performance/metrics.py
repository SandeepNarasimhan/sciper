from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def is_classifier(model):
    """Check if model is a classifier."""
    return (
        hasattr(model, "predict_proba")
        or hasattr(model, "predict")
        and hasattr(model, "classes_")
    )


def is_regressor(model):
    """Check if model is a regressor."""
    return hasattr(model, "predict") and not hasattr(model, "classes_")


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance automatically based on model type."""
    y_pred = model.predict(X_test)

    print("==== Model Evaluation ====")
    print(f"Model: {model.__class__.__name__}")

    if is_classifier(model):
        print("\nTask: Classification")
        print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
        print(
            f"Precision      : {precision_score(y_test, y_pred, average='weighted'):.4f}"
        )
        print(
            f"Recall         : {recall_score(y_test, y_pred, average='weighted'):.4f}"
        )
        print(f"F1 Score       : {f1_score(y_test, y_pred, average='weighted'):.4f}")

        try:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(y_test, y_proba[:, 1])
                print(f"ROC AUC        : {auc:.4f}")
        except Exception as e:
            print(e)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    elif is_regressor(model):
        print("\nTask: Regression")
        print(f"R² Score       : {r2_score(y_test, y_pred):.4f}")
        print(f"MAE            : {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE            : {mean_squared_error(y_test, y_pred):.4f}")
        print(f"RMSE           : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    else:
        print("Model type not recognized. Please check the model instance.")


def plot_model(model, X_test, y_test):
    """Plot performance metrics for classifier or regressor."""
    y_pred = model.predict(X_test)

    if is_classifier(model):
        print("🔍 Detected Task: Classification")

        # Confusion Matrix
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (only for binary or multilabel with predict_proba)
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    RocCurveDisplay.from_estimator(model, X_test, y_test)
                    plt.title("ROC Curve")
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            print(f"Could not plot ROC Curve: {e}")

        # Precision-Recall Curve
        try:
            if hasattr(model, "predict_proba"):
                PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
                plt.title("Precision-Recall Curve")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not plot PR Curve: {e}")

    elif is_regressor(model):
        print("🔍 Detected Task: Regression")

        # Actual vs Predicted
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
        plt.plot(
            [min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="red"
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.show()

        # Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30, color="purple")
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        # Residuals vs Predicted
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        plt.show()

    else:
        print("❗ Unrecognized model type. Cannot generate plots.")
