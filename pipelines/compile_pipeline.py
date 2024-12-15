import kfp
from kfp.compiler import Compiler
from pipelines.train_pipeline import iris_training_pipeline  # Adjust the import path as necessary

# Compile the pipeline
Compiler().compile(iris_training_pipeline, 'pipeline.json')

print("Pipeline compiled successfully into pipeline.json")
