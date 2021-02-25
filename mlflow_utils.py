
class MLflower():
    
    """
    A class to allow MLflow logging to active run
    Input categorical according dpp_report features are one-hot encoded and discretized features are labeled encoded
    
    :param ?: ?
    :type ?: ?
    
    """
    
    def __init__(self, active_run_id):
        
        self.active_run_id = active_run_id
    
    # def log_param_dict(param_dict):
    #     for param_k, param_v in param_dict.items():
    #         mlflow.log_metric(param_k, param_v)
     
    def log_param_dict(self,param_dict):
        import mlflow
        mlflow.log_params(param_dict)   
        
    def log_metric_dict(self,metric_dict):
        import mlflow
        mlflow.log_metrics(metric_dict, step = None) 
        