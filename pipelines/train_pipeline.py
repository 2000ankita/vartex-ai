from kfp import dsl
from kfp.dsl import component, Output, Dataset, Model, Input



@component
def preprocess_component(output_dir: Output[Dataset]):
    """
    Preprocess the Iris dataset and save the results to the output directory.
    """
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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
    train.to_csv(f"{output_dir.path}/train.csv", index=False)
    test.to_csv(f"{output_dir.path}/test.csv", index=False)


@component
def train_component(train_file: Input[Dataset], model_dir: Output[Model]):
    """
    Train a Random Forest Classifier on the preprocessed Iris dataset.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load training data
    train = pd.read_csv(train_file.path)
    X_train = train.iloc[:, :-1]
    y_train = train["target"]

    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{model_dir.path}/iris_model.pkl")


@dsl.pipeline(name="iris-training-pipeline", pipeline_root="gs://your-bucket-name/artifacts")
def iris_training_pipeline():
    # Preprocessing step
    preprocess_task = preprocess_component()

    # Training step
    train_task = train_component(
        train_file=preprocess_task.outputs["output_dir"]
    )
