import kfp
from kfp.compiler import Compiler
from train_pipeline import iris_training_pipeline  # Import the pipeline

# Compile the pipeline
Compiler().compile(iris_training_pipeline, 'pipeline.json')

print("Pipeline compiled successfully into pipeline.json")
