import logging
from mlflow.client import MlflowClient

class MlflowOperator:
    def __init__(self, endpoint):
        self.client = MlflowClient(tracking_uri=endpoint, registry_uri=endpoint)

    def get_latest_run_id(self, experiment_name):
        """
        Retrieve the latest run_id from the specified MLflow experiment.
        
        Args:
            experiment_name (str): Name of the MLflow experiment.
        
        Returns:
            str: The latest run_id.
        
        Raises:
            ValueError: If the experiment or runs are not found.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'.")
        
        return runs[0].info.run_id, experiment.experiment_id
    
    def get_latest_model_path(self, experiment_name, stage):
        """
        Retrieve the latest model version from MLflow model registry for a given experiment and stage.
        
        Args:
            experiment_name (str): Name of the MLflow experiment.
            stage (str): Model stage (e.g., 'Production', 'Staging', 'Archived').
            
        Returns:
            None: Prints and logs the model artifact path.
            
        Raises:
            Exception: If no model versions are found or if there's an error loading the model.
            
        Note:
            The function logs the model version and artifact path, and prints the artifact path to stdout.
            The artifact path is constructed as '{experiment_id-1}/{run_id}/artifacts'.
        """
        versions = self.client.search_model_versions(f"name='{experiment_name}'")
        production_versions = [v for v in versions if v.current_stage == stage]
        try:
            if versions:
                latest_version = max(production_versions, key=lambda v: int(v.version))
                run_id = latest_version.run_id
                run = self.client.get_run(run_id)
                experiment_id = run.info.experiment_id
                model_path = f"{int(experiment_id)-4}/{run_id}/artifacts/model/artifacts/BartPho_ViT_GPT2_LoRA_ICG_final"
                logging.info(f"Model version: {latest_version.version}")
                logging.info(f"MinIO path: {model_path}")
                print(f"Model path: {model_path}")
                return model_path
            else:
                raise Exception("No production model found.")
        except Exception as ex:
            raise Exception(f"Error in loading latest model --> {str(ex)}")

    def get_model_path(self, experiment_name):
        """
        Construct the MinIO path for the model artifact.
        
        Args:
            run_id (str): The MLflow run_id.
            bucket_name (str): The MinIO bucket name.
        
        Returns:
            str: The MinIO path to the model.
        """
        run_id, experiment_id = self.get_latest_run_id(experiment_name)
        return f"{int(experiment_id)-3}/{run_id}/artifacts"  # Adjust path based on your artifact structure
    
    
    
if __name__ == "__main__":
    mlflow_operator = MlflowOperator(endpoint="http://36.50.135.226:7893")
    model_path = mlflow_operator.get_latest_model_path("BartPho_ViT_GPT2_LoRA_ICG", "Production")
    print(f"{model_path}/model/artifacts/BartPho_ViT_GPT2_LoRA_ICG_final")

