---
title: MLOps Part 3 - Evaluation and Optimization
date: 2021-10-21 11:57:21
tags:
---

Now that we have trained a model and store it as a reusable artifact, we are ready to evaluate the model on unseen data. As with usual training practice, we are going to pull out the test portion of our split data, run this data through the trained model, and record the score we got from the test data. As a good measure, we will also re-run training process with mlflow-powered hyperparameter sweep and discover the most optimal hyperparameter that could gave us best generalization between training and testing data.

One important thing that is often overlooked when building a training pipeline is monitoring model drift and data drift. Therefore we are going to implement storing the distribution of our label and each of the features during training as a summary with the `r2` and `mae` performance metrics. When comparing two models, we can use this summary to compare the distribution of the data that is used to train each respective model.

![image](https://drive.google.com/uc?export=view&id=1KVqCU7TUzuln1CPufvR60X9_a4Evqf1r)

We're gonna finish the `evaluate` part or our pipeline. Let's go!

## Develop evaluate step

### Adding in logic

We need to add new function to our `BaseModel` object. Open `nyc_airbnb/utils/base_model.py` and add the following changes.

> ./nyc_airbnb/utils/base_model.py
> 
> ```python
> import wandb
> import logging
> import mlflow
> import pandas as pd
> import tempfile
> 
> logging.basicConfig(
>     level=logging.INFO,
>     format="%(asctime)-15s %(levelname)s - %(message)s")
> logger = logging.getLogger()
> 
> 
> class BaseRunner:
>     def __init__(self, wandb_run):
>         self.wandb_run = wandb_run
> 
>     def log_artifact(self,
>                      artifact_name: str,
>                      artifact_type: str,
>                      artifact_description: str,
>                      filename: str) -> wandb.Artifact:
>         """Log the provided local filename as an artifact in W&B, and add the artifact path
>         to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
>         Args:
>             artifact_name: name for the artifact
>             artifact_type:
>                 type for the artifact (just a string like "raw_data", "clean_data" and so on)
>             artifact_description: a brief description of the artifact
>             filename: local filename for the artifact
>         Returns:
>             Wandb artifact object
>         """
>         # Log to W&B
>         artifact = wandb.Artifact(
>             artifact_name,
>             type=artifact_type,
>             description=artifact_description,
>         )
>         artifact.add_file(filename)
>         self.wandb_run.log_artifact(artifact)
>         logger.info(f"Uploading {artifact_name} to Weights & Biases")
> 
>         # We need to call .wait() method to ensure that artifact transport has completed
>         # before we exit this method execution
>         if wandb.run.mode == 'online':
>             artifact.wait()
> 
>         return artifact
> 
>     def log_model(self,
>                   artifact_name: str,
>                   artifact_type: str,
>                   artifact_description: str,
>                   model_dir: str) -> wandb.Artifact:
>         """Log the provided local filename as an artifact in W&B, and add the artifact path
>         to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
>         Args:
>             artifact_name: name for the artifact
>             artifact_type:
>                 type for the artifact (just a string like "raw_data", "clean_data" and so on)
>             artifact_description: a brief description of the artifact
>             model_dir: local path for the model directory
>         Returns:
>             Wandb artifact object
>         """
>         # Log to W&B
>         artifact = wandb.Artifact(
>             artifact_name,
>             type=artifact_type,
>             description=artifact_description,
>         )
>         artifact.add_dir(model_dir)
>         self.wandb_run.log_artifact(artifact)
>         # We need to call .wait() method to ensure that artifact transport has completed
>         # before we exit this method execution
>         if wandb.run.mode == 'online':
>             artifact.wait()
> 
>         return artifact
> 
>     def retrieve_dataset_artifact(self, artifact_name) -> pd.DataFrame:
>         """Retrieve wandb artifact as pandas DataFrame, artifact_name should exist in
>         the context of current run. This function will only retrieve dataset artifact,
>         not model or any other artifact type.
>         Args:
>             artifact_name: name for the artifact
>         Returns:
>             DataFrame representation of the artifact
>         """
>         artifact_local_path = self.wandb_run.use_artifact(artifact_name).file()
> 
>         try:
>             data = pd.read_parquet(artifact_local_path)
>         except FileNotFoundError as err:
>             logger.error(f"{artifact_name} is not found")
>             raise err
> 
>         return data
> ```
> ```python
>     def retrieve_model(self, model_artifact):
>         """Retrieve model artifact from wandb.
> 
>         Args:
>             model_artifact: name for the artifact
>         Returns:
>             MLflow compatible model
>         """
>         logger.info("Downloading model")
>         # Download input artifact. This will also log that this script is using this
>         # particular version of the artifact
>         with tempfile.TemporaryDirectory() as temp_dir:
>             model_local_path = self.wandb_run.use_artifact(model_artifact) \
>                 .download(f'{temp_dir}/')
>             model = mlflow.sklearn.load_model(model_local_path)
>             return model
> ```

We can call this new function in evaluation entry point that we are going to build. But first, create a new file.

```
touch nyc_airbnb/evaluate.py
```

Now let's put in the logic for evaluation.

> ./nyc_airbnb/evaluate.py
> 
> ```python
> import argparse
> import logging
> import sys
> import tempfile
> 
> import mlflow
> import pandas as pd
> import wandb
> from sklearn.metrics import r2_score, mean_absolute_error
> 
> sys.path.append(".")
> from nyc_airbnb.utils.base_runner import BaseRunner
> 
> logging.basicConfig(
>     level=logging.INFO,
>     format="%(asctime)-15s %(levelname)s - %(message)s")
> logger = logging.getLogger()
> 
> 
> class EvaluateModelRunner(BaseRunner):
>     def __init__(self, wandb_run):
>         super().__init__(wandb_run)
> 
>     def test_data(self, data: pd.DataFrame, model, label: str):
>         # Read test dataset
>         X_test = data.copy()
>         y_test = X_test.pop(label)
> 
>         # Predict
>         y_pred = model.predict(X_test)
> 
>         # Evaluate
>         logger.info("Scoring")
>         r2 = r2_score(y_test, y_pred)
>         mae = mean_absolute_error(y_test, y_pred)
> 
>         logger.info(f"Score: {r2}")
>         logger.info(f"MAE: {mae}")
> 
>         # Log MAE and r2
>         self.wandb_run.summary['Test r2'] = r2
>         self.wandb_run.summary['Test mae'] = mae
> 
>         return y_pred, r2, mae
> 
> 
> if __name__ == "__main__":
>     # Process arguments
>     parser = argparse.ArgumentParser(description="Clean the training dataset")
>     parser.add_argument("test_model",
>                         type=str,
>                         help="Artifact name of trained model")
>     parser.add_argument("test_dataset",
>                         type=str,
>                         help="Artifact name of test dataset")
>     parser.add_argument("label",
>                         type=str,
>                         help="Label of test dataset")
>     args = parser.parse_args()
> 
>     wandb_run = wandb.init(job_type="test_model")
> 
>     runner = EvaluateModelRunner(wandb_run)
>     test_set = runner.retrieve_dataset_artifact(args.test_dataset)
>     model = runner.retrieve_model(args.test_model)
>     _ = runner.test_data(test_set, model, args.label)
> 
>     sys.exit(0)
> 
> ```

### Setting up entry point

We're half way through, let's open `MLproject` and add new entry point for evaluation there.

> ./MLproject
> 
> ```yaml
> name: nyc_airbnb
> conda_env: conda.yml
> 
> entry_points:
>   main:
>     parameters:
>       steps:
>         description: Comma-separated list of steps to execute (useful for debugging)
>         type: str
>         default: all
>       hydra_options:
>         description: Other configuration parameters to override
>         type: str
>         default: ''
>     command: "python main.py main.steps=\\'{steps}\\' $(echo {hydra_options})"
>   get_data:
>     parameters:
>       bucket:
>         description: OSS bucket where data is stored
>         type: string
>       object_path:
>         description: OSS object of dataset
>         type: string
>       artifact_name:
>         description: Name for the output artifact
>         type: string
>       artifact_type:
>         description: Type of the output artifact. This will be used to categorize the artifact in the W&B interface
>         type: string
>       artifact_description:
>         description: A brief description of the output artifact
>         type: string
>     command: "python nyc_airbnb/get_data.py
>         {bucket}
>         {object_path}
>         {artifact_name}
>         {artifact_type}
>         {artifact_description}"
>   split:
>     parameters:
>       input_artifact:
>         description: Artifact to split (a CSV file)
>         type: string
>       test_size:
>         description: Size of the test split. Fraction of the dataset, or number of items
>         type: string
>       random_seed:
>         description: Seed for the random number generator. Use this for reproducibility
>         type: string
>         default: 42
>       stratify_by:
>         description: Column to use for stratification (if any)
>         type: string
>         default: 'none'
>     command: "python nyc_airbnb/split_train_test.py
>         {input_artifact}
>         {test_size}
>         {random_seed}
>         {stratify_by}"
>   train:
>     parameters:
>       label:
>         description: Label column name
>         type: string
>       trainval_artifact:
>         description: Train dataset
>         type: string
>       val_size:
>         description: Size of the validation split. Fraction of the dataset, or number of items
>         type: string
>       random_seed:
>         description: Seed for the random number generator. Use this for reproducibility
>         type: string
>         default: 42
>       stratify_by:
>         description: Column to use for stratification (if any)
>         type: string
>         default: 'none'
>       rf_config:
>         description: Random forest configuration. A path to a JSON file with the configuration that will
>                     be passed to the scikit-learn constructor for RandomForestRegressor.
>         type: string
>       max_tfidf_features:
>         description: Maximum number of words to consider for the TFIDF
>         type: string
>       output_artifact:
>         description: Name for the output artifact
>         type: string
>     command: "python nyc_airbnb/train.py
>         {label}
>         {trainval_artifact}
>         {val_size}
>         {random_seed}
>         {stratify_by}
>         {rf_config}
>         {max_tfidf_features}
>         {output_artifact}"
> ```
> ```yaml
>   evaluate:
>     parameters:
>       test_model:
>         description: An MLflow serialized model
>         type: string
>       test_dataset:
>         description: The test artifact
>         type: string
>       label:
>         description: Label of the prediction
>         type: string
>     command: "python nyc_airbnb/evaluate.py
>         {test_model}
>         {test_dataset}
>         {label}"
> ```

Now we are going to call this new entry point from our `main.py`

> ./main.py
> 
> ```python
> import json
> import mlflow
> import tempfile
> import os
> import hydra
> from omegaconf import DictConfig
> from dotenv import load_dotenv
> 
> load_dotenv()
> ```
> ```python
> _steps = [
>     'download',
>     'split',
>     'train',
>     'evaluate'
> ]
> ```
> ``` python
> 
> 
> # This decorator automatically reads in the configuration
> @hydra.main(config_name='config')
> def go(config: DictConfig):
>     # Setup the wandb experiment. All runs will be grouped under this name
>     os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
>     os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
> 
>     # Steps to execute
>     steps_par = config['main']['steps']
>     active_steps = steps_par.split(",") if steps_par != "all" else _steps
> 
>     # Move to a temporary directory
>     with tempfile.TemporaryDirectory() as tmp_dir:
>         if "download" in active_steps:
>             # Download file and load in W&B
>             _ = mlflow.run(
>                 hydra.utils.get_original_cwd(),
>                 "get_data",
>                 parameters={
>                     "bucket": config["data"]["bucket"],
>                     "object_path": f"{config['data']['object']}",
>                     "artifact_name": f"{config['data']['raw_data']}",
>                     "artifact_type": "raw_data",
>                     "artifact_description": "Raw dataset from data store"
>                 }
>             )
> 
>         if "split" in active_steps:
>             _ = mlflow.run(
>                 hydra.utils.get_original_cwd(),
>                 "split",
>                 parameters={
>                     "input_artifact": f"{config['data']['raw_data']}:latest",
>                     "test_size": config['modeling']['test_size'],
>                     "random_seed": config['modeling']['random_seed'],
>                     "stratify_by": config['modeling']['stratify_by'],
>                 }
>             )
> 
>         if "train" in active_steps:
>             # NOTE: we need to serialize the random forest configuration into JSON
>             rf_config = os.path.abspath("rf_config.json")
>             with open(rf_config, "w+") as fp:
>                 json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH
>             # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
>             _ = mlflow.run(
>                 os.path.join(hydra.utils.get_original_cwd()),
>                 "train",
>                 parameters={
>                     "label": config["data"]["label"],
>                     "trainval_artifact": f"{config['data']['training_data']}:latest",
>                     "val_size": config["modeling"]["val_size"],
>                     "random_seed": config["modeling"]["random_seed"],
>                     "stratify_by": config["modeling"]["stratify_by"],
>                     "rf_config": rf_config,
>                     "max_tfidf_features": config["modeling"]["max_tfidf_features"],
>                     "output_artifact": config["main"]["experiment_name"],
>                 },
>             )
> ```
> ```python
>         if "evaluate" in active_steps:
>             _ = mlflow.run(
>                 os.path.join(hydra.utils.get_original_cwd()),
>                 "test",
>                 parameters={
>                     "test_model": f'{config["modeling"]["test_model"]}',
>                     "test_dataset": config['data']['test_data'],
>                     "label": config['data']['label_col'],
>                 },
>             )
> 
> 
> if __name__ == "__main__":
>     go()
> ```

Now that we can call `evaluate` from `main` entry point, we just need to add configuration that will be passed in to `evaluate` step.

> ./config.yaml
> 
> ```yaml
> main:
>   project_name: "nyc_airbnb"
>   experiment_name: "random_forest_model"
>   steps: all
> data:
>   bucket: "junda-mlops"
>   object: "dataset/training_data/"
>   raw_data: "raw_training_data.parquet"
>   training_data: "trainval_data.parquet"
>   test_data: "test_data.parquet"
>   label: "price"
> modeling:
>   # Name of exported model to be used in testing model
>   test_model: "random_forest_export:latest"
>   # Fraction of data to use for test (the remaining will be used for train and validation)
>   test_size: 0.2
>   # Fraction of remaining data to use for validation
>   val_size: 0.2
>   # Fix this for reproducibility, change to have new splits
>   random_seed: 42
>   # Column to use for stratification (use "none" for no stratification)
>   stratify_by: "none"
>   max_tfidf_features: 15
>   # NOTE: you can put here any parameter that is accepted by the constructor of
>   # RandomForestRegressor. This is a subsample, but more could be added:
>   random_forest:
>     n_estimators: 100
>     max_depth: 12
>     min_samples_split: 4
>     min_samples_leaf: 3
>     # Here -1 means all available cores
>     n_jobs: -1
>     criterion: mae
>     max_features: 0.5
>     # DO not change the following
>     oob_score: true
> ```

### Test run

Now we can run `evaluate` step. However, just for presentation purpose, let's run both `train` and `evaluate`.

```
mlflow run . -P steps=train,evaluate
```

If we open wandb dashboard now, we can see our previous training run without test r2 and test mae. WHile our most recent run has all score recorded for test and training.

![wandb screenshot](https://drive.google.com/uc?export=view&id=1WUKB5kZDA3Fxp55exCOA9nFBg2f-M9dV)

## Model retraining with hyperparameter sweep

## Adding in distribution into summary
