from google.cloud import aiplatform

def submit_pipeline():
    # Update these values
    PROJECT_ID = "corded-layout-443205-a8"  # Replace with your GCP project ID
    REGION = "us-central1"
    PIPELINE_TEMPLATE_PATH = "gs://bucketz1234nix/pipelines/pipeline.json"
    PIPELINE_ROOT = "gs://bucketz1234nix/artifacts/"
    DISPLAY_NAME = "example-pipeline-run"

    aiplatform.init(project=PROJECT_ID, location=REGION)

    pipeline_job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=PIPELINE_TEMPLATE_PATH,
        pipeline_root=PIPELINE_ROOT,
    )

    pipeline_job.run(sync=True)

if __name__ == "__main__":
    submit_pipeline()
