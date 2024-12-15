from kfp.v2 import dsl
from kfp.v2.dsl import component

@component
def preprocess_component(output_dir: str):
    import subprocess
    subprocess.run(["python", "components/preprocessing/preprocess.py", output_dir])

@component
def train_component(train_file: str, model_dir: str):
    import subprocess
    subprocess.run(["python", "components/training/train.py", train_file, model_dir])

@dsl.pipeline(name="iris-training-pipeline")
def iris_training_pipeline(pipeline_root: str):
    # Preprocessing
    preprocess_task = preprocess_component(output_dir=f"{pipeline_root}/data")
    
    # Training
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
