from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

# Preprocessing component
@component
def preprocess_component(output_dir: Output[Dataset]):
    import subprocess
    subprocess.run(["python", "components/preprocessing/preprocess.py", output_dir])

# Training component
@component
def train_component(train_file: Input[Dataset], model_dir: Output[Model]):
    import subprocess
    subprocess.run(["python", "components/training/train.py", train_file, model_dir])

# Pipeline definition
@dsl.pipeline(name="iris-training-pipeline")
def iris_training_pipeline(pipeline_root: str):
    # Preprocessing step
    preprocess_task = preprocess_component(output_dir=f"{pipeline_root}/data")

    # Training step
    train_task = train_component(
        train_file=preprocess_task.outputs["output_dir"] + "/train.csv",
        model_dir=f"{pipeline_root}/model"
    )

if __name__ == "__main__":
    from kfp.v2.compiler import Compiler
    Compiler().compile(
        pipeline_func=iris_training_pipeline,
        package_path="pipeline.json"
    )
