import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod


def is_classifier(model):
    """Check if model is a classifier."""
    return (
        hasattr(model, "predict_proba")
        or (hasattr(model, "predict") and hasattr(model, "classes_"))
        or (isinstance(model, BaseEstimator) and hasattr(model, '_estimator_type') and model._estimator_type == 'classifier')
    )


def is_regressor(model):
    """Check if model is a regressor."""
    return (
        (hasattr(model, "predict") and not hasattr(model, "classes_"))
        or (isinstance(model, BaseEstimator) and hasattr(model, '_estimator_type') and model._estimator_type == 'regressor')
    )

class Explainer(ABC):
    """Abstract base class for explainers."""
    @abstractmethod
    def explain(self, model, X_train, X_test, index=0):
        pass



def explain_model(model, X_train, X_test, index=0):
    """
    Generates SHAP and LIME explanations automatically based on model type.

    Parameters:
        model: trained model
        X_train: pandas DataFrame of training data (used to fit explainer)
        X_test: pandas DataFrame of test data
        index: index of instance in X_test to explain with LIME
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_train and X_test must be pandas DataFrames.")

    print("🔍 Automatically detecting model type...")
    if is_classifier(model):
        task = "classification"
        print("✅ Detected as classification model")
    elif is_regressor(model):
        task = "regression"
        print("✅ Detected as regression model")
    else:
        raise ValueError("❌ Could not detect model type.")

    # Basic handling for feature importances for tree-based models
    if hasattr(model, 'feature_importances_'):
        print("\n📊 Feature Importances (Tree-based Models)")
        try:
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            importances.sort_values(ascending=False, inplace=True)
            importances.plot(kind='bar')
            plt.title("Feature Importances")
            plt.ylabel("Importance")
            plt.show()
        except Exception as e:
            print(f"❌ Feature importances plotting failed: {e}")

    # Use the Explainer interface for SHAP and LIME
    shap_explainer = SHAPExplainer()
    shap_explainer.explain(model, X_train, X_test)

    lime_explainer = LIMEExplainer()
    lime_explainer.explain(model, X_train, X_test, index)

    # SHAP
    print("\n🧠 SHAP Summary Plot (Global Interpretation)")
    try:
        shap_explainer = shap.Explainer(model, X_train)
        shap_values = shap_explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=True)
    except Exception as e:
        print(f"❌ SHAP failed: {e}")

    # LIME
    print("\n🎯 LIME Explanation (Instance: index = {})".format(index))
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            mode=task,
            class_names=(np.unique(model.predict(X_train)).astype(str).tolist() if task == "classification" else None)
        )

        instance = X_test.iloc[index].values

        lime_exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba if task == "classification" else model.predict
        )

        lime_exp.show_in_notebook()
        # Or use: lime_exp.as_pyplot_figure() if outside notebook
    except Exception as e:
        print(f"❌ LIME failed: {e}")
