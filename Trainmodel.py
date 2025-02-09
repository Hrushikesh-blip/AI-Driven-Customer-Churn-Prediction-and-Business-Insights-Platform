from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import pandas as pd

def train_model(data_path, model_path):
    """Train a churn prediction model."""
    data = pd.read_csv(data_path)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train_model("data/processed/cleaned_data.csv", "models/churn_model.pkl")
