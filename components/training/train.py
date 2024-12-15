import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(train_file: str, model_dir: str):
    # Load training data
    train = pd.read_csv(train_file)
    X_train = train.iloc[:, :-1]
    y_train = train["target"]

    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{model_dir}/iris_model.pkl")

if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])
