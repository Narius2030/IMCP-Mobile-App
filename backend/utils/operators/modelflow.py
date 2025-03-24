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
        return f"{int(experiment_id)-1}/{run_id}/artifacts"  # Adjust path based on your artifact structure
    
    
    
if __name__ == "__main__":
    mlflow_operator = MlflowOperator(endpoint="http://160.191.244.13:7893")
    print(mlflow_operator.get_model_path("Image_Captioning"))

