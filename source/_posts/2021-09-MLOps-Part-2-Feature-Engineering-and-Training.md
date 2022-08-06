---
title: MLOps Part 2 - Feature Engineering and Training
date: 2021-09-14 23:00:18
tags:
---
 
Previously, we have set up the main skeleton of our training pipeline using mlflow project and implemented a `download` step component. Now let's continue building the training pipeline.
 
![image](https://drive.google.com/uc?export=view&id=1KVqCU7TUzuln1CPufvR60X9_a4Evqf1r)
 
Right now we are going to develop the feature engineering and training part. For the sake of simplicity, we are going to implement a bare minimum feature engineering for our model, because we are looking to focus our work on mlops. It is very possible to develop a more rigorous feature engineering step that results in much better model performance.
 
<!-- more -->
 
## Develop split step
 
Let's start with creating a new file that will split our downloaded data on wandb into a training part and testing part.
 
```
touch nyc_airbnb/split_train_test.py
```
 
Just like `nyc_airbnb/get_data.py`, here we will build the logic to split our full dataset into a training and testing set. We do this early to safeguard our model from information leakage. Open the new file and put the following code into it.
 
> ./nyc_airbnb/split_train_test.py
> 
> ```python
> import argparse
> import logging
> import sys
> import os
> import tempfile
>  
> import wandb
> import pandas as pd
> from sklearn.model_selection import train_test_split
>  
> sys.path.append(".")
> from nyc_airbnb.utils.base_runner import BaseRunner
>  
> logging.basicConfig(
>    level=logging.INFO,
>    format="%(asctime)-15s %(levelname)s - %(message)s")
> logger = logging.getLogger()
>  
>  
> class SplitRunner(BaseRunner):
>    def __init__(self,
>                 wandb_run,
>                 test_size,
>                 random_seed,
>                 stratify_by):
>        super().__init__(wandb_run)
>        self.test_size = test_size
>        self.random_seed = random_seed
>        self.stratify_by = stratify_by
>  
>    def split_train_test(self,
>                         data: pd.DataFrame,
>                         dir_name: str):
>        trainval, test = train_test_split(
>            data,
>            test_size=float(self.test_size),
>            random_state=int(self.random_seed),
>            stratify=data[self.stratify_by] if self.stratify_by != 'none' else None,
>        )
>  
>        logger.info(f'train proportion contains {trainval.shape[0]}')
>        logger.info(f'test proportion contains {test.shape[0]}')
>  
>        file_dict = {}
>        for data_frame, name in zip([trainval, test], ['trainval', 'test']):
>            logger.info(f"Uploading {name}_data.parquet dataset")
>            temp_file = os.path.join(dir_name, f'{name}_data.parquet')
>            data_frame.to_parquet(
>                temp_file,
>                index=False,
>                engine='pyarrow',
>                compression='gzip')
>            file_dict[f'{name}_data'] = temp_file 
>  
>        return file_dict
>  
>  
> if __name__ == "__main__":
>    parser = argparse.ArgumentParser(description="Split training dataset")
>    parser.add_argument("input_artifact",
>                        type=str,
>                        help="Reference to mlflow artifact of input data")
>    parser.add_argument("test_size",
>                        type=str,
>                        help="The size of test data")
>    parser.add_argument("random_seed",
>                        type=str,
>                        help="Random seed")
>    parser.add_argument("stratify_by",
>                        type=str,
>                        help="Column to use for stratification")
>    args = parser.parse_args()
>  
>    runner = SplitRunner(
>        wandb.init(job_type="split_data"),
>        args.test_size,
>        args.random_seed,
>        args.stratify_by
>    )
>    dataset = runner.retrieve_dataset_artifact(args.input_artifact)
>  
>    with tempfile.TemporaryDirectory() as temp_dir:
>        files = runner.split_train_test(dataset, temp_dir)
>        for key, file_name in files.items():
>            _ = runner.log_artifact(
>                f'{key}.parquet',
>                key,
>                f'{key} split of the dataset',
>                file_name
>            )
>  
>    sys.exit(0)
> ```
 
The logic here is quite simple, we listen for arguments of which artifact name from wandb we need to process. The actual artifact name will be supplied by the main entry point. We then download it, split it using sklearn's `train_test_split`, and the result test and train split is logged back to wandb. The name of the artifact for both training portion and testing portion is also configurable, so that main entry point can further pass it on to the next component.
 
Now we have to strap this into `MLproject`.
 
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
> ```
> ```yaml
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
> ```
 
With `MLproject` entry point set, we can now call this from `main.py` with values retrieved from `config.yaml`.
 

> ./main.py
> 
> ```python
> import json
> import mlflow
> import tempfile
> import os
> import wandb
> import hydra
> from omegaconf import DictConfig
> from dotenv import load_dotenv
> 
> load_dotenv()
> ```
> ```python
> _steps = [
>     'download',
>     'split'
> ]
> ```
> ```python
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
> ```
>
> ```python
>         if "split" in active_steps:
>             _ = mlflow.run(
>                 hydra.utils.get_original_cwd(),
>                 "split",
>                 parameters={
>                     "input_artifact": f"{config['data']['raw_data']}",
>                     "test_size": config['modeling']['test_size'],
>                     "random_seed": config['modeling']['random_seed'],
>                     "stratify_by": config['modeling']['stratify_by'],
>                 }
>             )
> 
> 
> if __name__ == "__main__":
>     go()
> ```
 
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
> ```
> ```yaml
> modeling:
>   # Fraction of data to use for test (the remaining will be used for train and validation)
>   test_size: 0.2
>   # Fraction of remaining data to use for validation
>   val_size: 0.2
>   # Fix this for reproducibility, change to have new splits
>   random_seed: 42
>   # Column to use for stratification (use "none" for no stratification)
>   stratify_by: "none"
> ```
 
With this we have set up the split step and we can test run this. We can either run entire pipeline, or just the `split` part. This is made possible because we pass `raw_data.parquet:latest` as an argument to `split` entry point. The `:latest` tag means we will always retrieve the most recent version of the file. If we have multiple version of our wadnb artifact, we can also pass the specific version as tag, such as `v0` for the first ever logged version of that particular artifact, or `v1` for the second version, etc.
 
To run only this component step, use ```mlflow run . -P steps=split```. To run the entire pipeline, the command is ```mlflow run .```.
 
After the run is finished, check our wandb dashboard and see if a new artifact called `trainval_data` and `test_data` exists. Both of these will be passed to subsequent steps.

![wandb dasboard](https://drive.google.com/uc?export=view&id=17U78FW1bPP-wX-d1ipqczr-_AWxOz9T_)
 
## Develop train_model step
 
Let's put in the logic for training our model. Create a new file for this.
 
```shell
touch nyc_airbnb/train.py
```
 
And paste in the following code.
 
> ./nyc_airbnb/train.py
> 
> ```python
> import argparse
> import json
> import logging
> import os
> import shutil
> import sys
> from typing import Dict
> 
> import mlflow
> import pandas as pd
> import wandb
> from sklearn.metrics import r2_score, mean_absolute_error
> from sklearn.model_selection import train_test_split
> 
> sys.path.append(".")
> from nyc_airbnb.utils.base_runner import BaseRunner
> from nyc_airbnb.utils.pipeline import get_inference_pipeline
> 
> logging.basicConfig(
>     level=logging.INFO,
>     format="%(asctime)-15s %(levelname)s - %(message)s")
> logger = logging.getLogger()
> 
> 
> class TrainModelRunner(BaseRunner):
>     def __init__(self,
>                  wandb_run,
>                  label,
>                  random_seed,
>                  stratify_by,
>                  val_size
>                  ):
>         super().__init__(wandb_run)
>         self.label = label
>         self.val_size = val_size
>         self.stratify_by = stratify_by
>         self.random_seed = random_seed
> 
>     def train(self,
>               data: pd.DataFrame,
>               rf_config: Dict[str, float],
>               max_tfidf_features):
>         """
>         Train a model by running fit() method of pipeline.
>         Args:
>             data: A DataFrame of training dataset
>             rf_config:
>                 A configuration dict to be used as model parameters
>             max_tfidf_features:
>                 hyper-param for tfidf preprocessor
>         """
>         y = data[self.label]
>         X = data.drop(self.label, axis=1)
> 
>         X_train, X_val, y_train, y_val = train_test_split(
>             X,
>             y,
>             test_size=self.val_size,
>             stratify=X[self.stratify_by],
>             random_state=self.random_seed
>         )
> 
>         logger.info("Preparing pipeline")
>         lasso_pipe = get_inference_pipeline(
>             rf_config,
>             max_tfidf_features
>         )
> 
>         logger.info("Training model")
>         trained_model = lasso_pipe.fit(X_train, y_train)
> 
>         # training performance metric
>         y_pred = trained_model.predict(X)
>         r2 = r2_score(y, y_pred)
>         mae = mean_absolute_error(y, y_pred)
> 
>         return r2, mae, trained_model
> 
>     def persist_model(self, model, model_artifact_name: str):
>         persist_dir = f'{model_artifact_name}_dir'
> 
>         # Remove if exists
>         if os.path.exists(persist_dir):
>             shutil.rmtree(persist_dir)
> 
>         mlflow.sklearn.save_model(
>             model,
>             persist_dir,
>         )
> 
>         self.log_model(
>             model_artifact_name,
>             "model_export",
>             "Pytorch lasso model export",
>             persist_dir)
> 
> 
> if __name__ == "__main__":
>     # Process arguments
>     parser = argparse.ArgumentParser(description="Train the model")
> 
>     parser.add_argument(
>         "trainval_artifact",
>         type=str,
>         help="Artifact containing the training dataset. It will be split into train and validation"
>     )
>     parser.add_argument(
>         "val_size",
>         type=float,
>         help="Size of the validation split. Fraction of the dataset, or number of items",
>     )
>     parser.add_argument(
>         "random_seed",
>         type=int,
>         help="Seed for random number generator",
>         default=42
>     )
>     parser.add_argument(
>         "stratify_by",
>         type=str,
>         help="Column to use for stratification",
>         default="none"
>     )
>     parser.add_argument(
>         "rf_config",
>         help="Random forest configuration. A JSON dict that will be passed to the "
>              "scikit-learn constructor for RandomForestRegressor.",
>         default="{}",
>     )
>     parser.add_argument(
>         "max_tfidf_features",
>         help="Maximum number of words to consider for the TFIDF",
>         default=10,
>         type=int
>     )
>     parser.add_argument(
>         "output_artifact",
>         type=str,
>         help="Name for the output serialized model"
>     )
>     args = parser.parse_args()
> 
>     wandb_run = wandb.init(job_type="training_model")
> 
>     with open(args.rf_config) as fp:
>         rf_config = json.load(fp)
> 
>     # Log model config to wandb runs
>     wandb_run.config.update(rf_config)
> 
>     # Run training
>     runner = TrainModelRunner(
>         wandb_run,
>         label=args.label
>     )
>     training_set = runner.retrieve_dataset_artifact(args.train_artifact)
>     r2, mae, TRAINED_MODEL = runner.train(training_set, rf_config, args.max_tfidf_features)
> 
>     # Logging to wandb
>     logger.info(f'R2 score is {r2}')
>     logger.info(f'MAE loss is {mae}')
>     wandb_run.summary['Training r2'] = r2
>     wandb_run.log({
>         "Training mae": mae,
>         "Training r2": r2
>     })
> 
>     # Persist model
>     logger.info('Exporting model')
>     runner.persist_model(TRAINED_MODEL, args.output_artifact)
> 
>     sys.exit(0)
> ```
 
In this step, we are going to train our model on the training split of the data. We score the performance on the model on *training* data, and log the trained model artifact and the performance into wandb. Later on, we are going use the same trained model artfact to score to *testing* data. Doing this gave us several advantage:
 
1. It safeguards us from data leakage. Remember that we split our dataset immediately after retrieving them from source, any cleaning/preprocessing we do in this step will only be applied to training split of the data.
2. Scoring separately allows us to safeguard from overfitting. We might apply cross-validation in our training step to discover the most optimal combination of hyperparameters and preprocessing logic, and the same hyperparameter and preprocessing will be applied upon testing data that we haven't seen at all during training. This will allow us to prepare our model to deal with new data in production.
 
We are also going to organise the source code in a way that is easy to maintain and extend. Here we are only going to retrieve training data and slap it into the training pipeline. The actual training and preprocessing logic will be built on different modules. This way, we may plug-in and plug-out any preprocessing logic or even swap out the model from logistic regression into gradient boosting, the training function here won't care as long as it can get the pipeline object.
 
To achieve this, we need to create the `get_inference_pipeline` function; it will return a scikit-learn compatible `Pipeline` object. `Pipeline` can assemble multiple estimators and transformers together. If we need to apply `OneHotEncoding`, `StandardScaler` and `RandomForest` together, we put them into a `Pipeline` and we can all `fit` only once from the `Pipeline`. The `Pipeline` will be the ones handling `fit` execution in sequence. When we want to save these fitted estimators together, we just need to save the `Pipeline`.
 
Create the module to store our `Pipeline` logic.
 
```
touch nyc_airbnb/utils/pipeline.py
```
 
Write the following code into the new file.
 
> ./nyc_airbnb/utils/pipeline.py
> 
> ```python
> import logging
> import numpy as np
> import pandas as pd
> from sklearn.compose import ColumnTransformer
> from sklearn.feature_extraction.text import TfidfVectorizer
> from sklearn.impute import SimpleImputer
> from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
> from sklearn.ensemble import RandomForestRegressor
> from sklearn.pipeline import Pipeline, make_pipeline
> 
> 
> def delta_date_feature(dates):
>     """
>     Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
>     between each date and the most recent date in its column
>     """
>     date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
>     return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()
> 
> 
> logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
> logger = logging.getLogger()
> 
> 
> def get_inference_pipeline(rf_config, max_tfidf_features):
>     # Let's handle the categorical features first
>     # Ordinal categorical are categorical values for which the order is meaningful, for example
>     # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
>     ordinal_categorical = ["room_type"]
>     non_ordinal_categorical = ["neighbourhood_group"]
>     # NOTE: we do not need to impute room_type because the type of the room
>     # is mandatory on the websites, so missing values are not possible in production
>     # (nor during training). That is not true for neighbourhood_group
>     ordinal_categorical_preproc = OrdinalEncoder()
> 
>     ######################################
>     # Build a pipeline with two steps:
>     # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
>     # 2 - A OneHotEncoder() step to encode the variable
>     non_ordinal_categorical_preproc = make_pipeline(
>         SimpleImputer(strategy="most_frequent"),
>         OneHotEncoder()
>     )
>     ######################################
> 
>     # Let's impute the numerical columns to make sure we can handle missing values
>     # (note that we do not scale because the RF algorithm does not need that)
>     zero_imputed = [
>         "minimum_nights",
>         "number_of_reviews",
>         "reviews_per_month",
>         "calculated_host_listings_count",
>         "availability_365",
>         "longitude",
>         "latitude"
>     ]
>     zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
> 
>     # A MINIMAL FEATURE ENGINEERING step:
>     # we create a feature that represents the number of days passed since the last review
>     # First we impute the missing review date with an old date (because there hasn't been
>     # a review for a long time), and then we create a new feature from it,
>     date_imputer = make_pipeline(
>         SimpleImputer(missing_values=None, strategy='constant', fill_value='2010-01-01'),
>         FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
>     )
> 
>     # Some minimal NLP for the "name" column
>     reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
>     name_tfidf = make_pipeline(
>         SimpleImputer(missing_values=None, strategy="constant", fill_value="imputed name"),
>         reshape_to_1d,
>         TfidfVectorizer(
>             binary=False,
>             max_features=max_tfidf_features,
>             stop_words='english'
>         )
>     )
> 
>     # Let's put everything together
>     preprocessor = ColumnTransformer(
>         transformers=[
>             ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
>             ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
>             ("impute_zero", zero_imputer, zero_imputed),
>             ("transform_date", date_imputer, ["last_review"]),
>             ("transform_name", name_tfidf, ["name"])
>         ],
>         remainder="drop",  # This drops the columns that we do not transform
>     )
> 
>     processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]
> 
>     # Create random forest
>     random_forest = RandomForestRegressor(**rf_config)
> 
>     ######################################
>     # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
>     # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
>     # with the random forest instance that we just saved in the `random_forest` variable.
>     # HINT: Use the explicit Pipeline instead of make_pipeline constructor to leverage named step
>     sk_pipe = Pipeline(
>         steps=[
>             ("preprocessor", preprocessor),
>             ("random_forest", random_forest)
>         ]
>     )
> 
>     return sk_pipe, processed_features
> 
> ```
 
Here we declare all preprocessing logic for each data type, ordinal categorical, nominal categorical, numeric, temporal, and textual. We strap all of these with `ColumnTransformer`, and the resulting transformer can be put into a `Pipeline` step. The final pipeline just needs to combine the `preprocessor` step and `random_forest` model.
 
As we can see, we build the steps in a declarative way, so we can change any part of the steps and it can be called by our `train` entry point as long as we return `Pipeline` from here.
 
Now that we have created the entry point for training, we just have to call it from our main entry point. Let's start by making the entry point for training in our MLproject.
 
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
> ```
> ```yaml
>     train:
>       parameters:
>         label:
>           description: Label column name
>           type: string
>         trainval_artifact:
>           description: Train dataset
>           type: string
>         val_size:
>           description: Size of the validation split. Fraction of the dataset, or number of items
>           type: string
>         random_seed:
>           description: Seed for the random number generator. Use this for reproducibility
>           type: string
>           default: 42
>         stratify_by:
>           description: Column to use for stratification (if any)
>           type: string
>           default: 'none'
>         rf_config:
>           description: Random forest configuration. A path to a JSON file with the configuration that will
>                       be passed to the scikit-learn constructor for RandomForestRegressor.
>           type: string
>         max_tfidf_features:
>           description: Maximum number of words to consider for the TFIDF
>           type: string
>         output_artifact:
>           description: Name for the output artifact
>           type: string
>       command: "python nyc_airbnb/train.py
>           {label}
>           {trainval_artifact}
>           {val_size}
>           {random_seed}
>           {stratify_by}
>           {rf_config}
>           {max_tfidf_features}
>           {output_artifact}"
> ```
 
Now we can call this from `main.py`.
 
> ./main.py
> 
> ```python
> import json
> import mlflow
> import tempfile
> import os
> import wandb
> import hydra
> from omegaconf import DictConfig
> from dotenv import load_dotenv
> 
> load_dotenv()
> ```
> ```python
> _steps = [
>     'download',
>     'split'
> ]
> ```
> ```python
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
>                     "input_artifact": f"{config['data']['raw_data']}",
>                     "test_size": config['modeling']['test_size'],
>                     "random_seed": config['modeling']['random_seed'],
>                     "stratify_by": config['modeling']['stratify_by'],
>                 }
>             )
> ```
>
> ```python
>          if "train" in active_steps:
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
>                     "output_artifact": "random_forest_export",
>                 },
>             )
> 
> 
> if __name__ == "__main__":
>     go()
> ```
 
Before we finish the training part and test run this, we need to add the configuration in `config.yaml`.
 
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
>   label: "price"
> modeling:
>   # Fraction of data to use for test (the remaining will be used for train and validation)
>   test_size: 0.2
>   # Fraction of remaining data to use for validation
>   val_size: 0.2
>   # Fix this for reproducibility, change to have new splits
>   random_seed: 42
>   # Column to use for stratification (use "none" for no stratification)
>   stratify_by: "none"
> ```
> ```yaml
>   max_tfidf_features: 15
>   # NOTE: you can put here any parameter that is accepted by the constructor of
>   # RandomForestRegressor. This is a subsample, but more could be added:
>   random_forest:
>     n_estimators: 100
>     max_depth: 15
>     min_samples_split: 4
>     min_samples_leaf: 3
>     # Here -1 means all available cores
>     n_jobs: -1
>     criterion: mae
>     max_features: 0.5
>     # DO not change the following
>     oob_score: true
> ```
 
Now we have everything ready to test run our training step. Same as previous example, we can run using mlflow cli.
 
```shell
mlflow run . -P steps=train
```

If we look at wandb.ai run summary, we are going to see training score of our model.

![wandb ss](https://drive.google.com/uc?export=view&id=1TvTLncVIr8iu-XjC3WlMefNUEK3EyNhZ)

Not only that, we can also see the hyper parameter we use to get that score, the datasets we consume, and their respective version. This is going to be very useful when we want to do offline evaluation of our model.

The progress up to this point can be accessed from the [accompanying repository](https://github.com/iahsanujunda/nyc_airbnb_pipeline) on `part-2` branch.
