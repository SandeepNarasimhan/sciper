from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from performance.metrics import evaluate_model, plot_model

# Load sample data
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X[:100,], y[:100], test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
evaluate_model(clf, X_test, y_test)
plot_model(clf, X_test, y_test)

explain_model()