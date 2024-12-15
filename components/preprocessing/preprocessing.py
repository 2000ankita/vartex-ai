import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(output_dir: str):
    # Load Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    # Split into train and test
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
    test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])

    # Save preprocessed data
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1])
