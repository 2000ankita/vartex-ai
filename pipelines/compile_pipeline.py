import sys
import os

# Add the parent directory of this file to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your pipeline
from pipelines.train_pipeline import iris_training_pipeline  # Adjust the import path as necessary

from kfp.compiler import Compiler

# Compile the pipeline
Compiler().compile(iris_training_pipeline, 'pipeline.json')

print("Pipeline compiled successfully into pipeline.json")
