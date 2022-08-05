---
title: MLOps Part 1 - Intro to MLflow Project and Setting-up Our First Component
date: 2021-08-02 21:49:38
tags:
---
 
MLflow is a very nice tool to handle our MLOps needs. It covers several important features for doing MLOps, namely tracking server, model registry, and source code packaging. Here we are going to focus on [MLFlow Projects](https://mlflow.org/docs/latest/projects.html), the source code packaging feature that can help us develop a reproducible machine learning pipeline.
 
MLFlow projects enable us to run source codes in a consistent way by encapsulating the runtime environment together with the source code, so that we can develop our source code on OSX, and have it run on linux with the same reproducible result, if we so need.
 
<!-- more -->
 
Let's imagine we build a machine learning training pipeline with the following function sequence.
 
![image](https://drive.google.com/uc?export=view&id=1KVqCU7TUzuln1CPufvR60X9_a4Evqf1r)
 
Leveraging the mlflow project will enable us to develop and run each component as a separate module. When we need to pass output of one component as an input of the succeeding component, we can do this as an argument.
 
I say, it's enough talking and let's start making!
 
## Implementation
 
To start building a pipeline using the mlflow project, we need to install the `mlflow` library. This can easily be done using `pip install mlflow`. This will be useful to test run our pipeline code later. For the sake of clarity, I will build a model to predict AirBnB rental prices in NYC.
 
### Initial setup
 
Let's start by making a new directory, and change into it.
 
```
mkdir rental_prediction_pipeline
cd rental_prediction_pipeline
```
 
To start building with the mlflow project, we need to create several mandatory files.
 
```
touch conda.yml MLproject
```
 
The `MLproject` file dictates how mlflow should run our source code. While `conda.yml` will be used to define our python environment to run the code.
 
For our needs, let's fill the `MLproject` file with the following configuration.
 
```
# ./MLProject
name: nyc_airbnb
conda_env: conda.yml
 
entry_points:
 main:
   parameters:
     steps:
       description: Comma-separated list of steps to execute (useful for debugging)
       type: str
       default: all
     hydra_options:
       description: Other configuration parameters to override
       type: str
       default: ''
   command: "python nyc_airbnb/main.py main.steps=\\'{steps}\\' $(echo {hydra_options})"
```
 
Let's go through this.
 
We first define the name of the environment we are running our source code in, called `nyc_airbnb`. Then we specify the file that contains how to create this environment in the `conda_env` part, with python it is common to use conda environment, therefore we provide our `conda.yml` into this. This file is currently empty but we will put something later.
 
After that we specify entry points. Mlflow asks that every `MLproject` should contain a `main` entry point. When we run our code with mlflow, it will start by triggering the `nyc_airbnb/main.py`.
 
Now, let's write into `conda.yml`.
 
```
# ./conda.yml
name: nyc_airbnb
channels:
 - conda-forge
 - defaults
dependencies:
 - python=3.8.12
 - autopep8=1.5.7
 - click=8.0.3
 - oss2=2.15.0
 - numpy=1.21.2
 - pandas=1.3.3
 - coverage=6.0
 - python-dotenv=0.19.0
 - hydra-core=1.0.6
 - pip=21.2.4
 - python-decouple
 - pyarrow=6.0.1
 - scikit-learn=0.24.1
 - scipy=1.7.3
 - tqdm=4.62.3
 - imbalanced-learn=0.8.1
 - wandb=0.12.1
 - pytorch=1.10.0
 - mlflow=1.20.2
 - pip:
     - fsspec==2021.11.0
```
 
Mlflow will create this python virtual environment before it starts to run our source code, it will only create a new environment once to speed-up subsequent processes. An exception is when we modify the dependecy verison or add a new dependency, mlflow will create a new environment so that the source code starts fresh.
 
For local development purposes, we should also create this virtual enviroment in our machine.
 
```
conda env create -f environment.yml
conda activate nyc_airbnb
```
 
### Develop main entry point
 
Now we can start to create our main entry point by making the directories.
 
```
mkdir nyc_airbnb
touch nyc_airbnb/main.py
```
 
Let's open the `nyc_airbnb/main.py` file, and write code in it.
 
```
# nyc_airbnb/main.py
import json
 
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
 
_steps = [
   "download"
]
 
 
# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
   # Setup the wandb experiment. All runs will be grouped under this name
   os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
   os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
 
   # Steps to execute
   steps_par = config['main']['steps']
   active_steps = steps_par.split(",") if steps_par != "all" else _steps
 
   # Move to a temporary directory
   with tempfile.TemporaryDirectory() as tmp_dir:
 
       if "download" in active_steps:
           # Download file and load in W&B
           pass
 
if __name__ == "__main__":
   go()
```
 
There is a lot to digest here so please bear with me.
 
We start by declaring our execution step as a list `_steps`. We start with only `download` in there, but we are going to grow it as we progress.
 
After that, we create a wrapper function called `go()` where we are going to put the execution logic. We use `hydra` to handle our configuration. All of our config will be declared inside a separate `config.yaml` file. Hydra needs to be tied-in to `go()` function as a decorator.
 
We start the function by setting up the environment variable, we read in the config from hydra and put it into the system's env variable.
 
After that we prepare `active_steps` - actual steps to execute. currently we only have one step so this part doesn't do much, but once we have multiple steps, it will allow us to override execution of `_steps` by running this form CLI
 
`mlflow run nyc_airbnb/ -P steps=train`
 
if we run without `steps` argument like this `mlflow run nyc_airbnb`, the list we have in `_steps` will be used instead.
 
Next part is where we put the skeleton that is going to tell mlflow what to run if `download` is in the steps that we should run.
 
### Develop get_data step
 
We need a way to digest the data from a source system into our training pipeline. Source systems can be a data warehouse, a flat file on a shared file system, or by scrapping a web page. Once we obtain this data, we want to log this data so that we can later revisit which data results in which model. For our case, the data is obtained from Alibaba Cloud OSS. If you are familiar with AWS S3, OSS is what S3 in alibaba cloud platform is called.
 
To start, let's create the file to hold our download logic.
 
```touch nyc_airbnb/get_data.py```
 
Open this file and let's put in our logic there.
 
```
# ./nyc_airbnb/get_data.py
 
import argparse
import logging
import os
import sys
import tempfile
import oss2
import pandas as pd
import wandb
 
sys.path.append("..")
from nyc_airbnb.utils.base_runner import BaseRunner
 
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.info(sys.path)
 
 
class GetOSSDataRunner(BaseRunner):
   def __init__(self,
                wandb_run,
                artifact_name,
                artifact_type,
                artifact_description):
       super().__init__(wandb_run)
       self.artifact_name = artifact_name
       self.artifact_type = artifact_type
       self.artifact_description = artifact_description
 
   def get_oss_data(self,
                    bucket,
                    object_path,
                    local_directory):
       self.wandb_run.config.update({
           'bucket': bucket,
           'object-path': object_path
       })
 
       # Setup OSS Connection
       logger.info("Connecting to Aliyun")
       auth = oss2.Auth(
           os.environ['OSS_ACCESS_KEY_ID'],
           os.environ['OSS_ACCESS_KEY_SECRET']
       )
       bucket = oss2.Bucket(
           auth,
           'https://oss-ap-southeast-5.aliyuncs.com',
           bucket
       )
 
       object_list = []
       for obj in oss2.ObjectIteratorV2(bucket, prefix=object_path):
           object_list.append(obj.key)
 
       # Check exported file in OSS
       try:
           assert len(object_list) <= 2
       except AssertionError as err:
           logger.error('Expect OSS path to contain only 1 file')
           raise err
 
       object_key = object_list[-1]
 
       logger.info("Downloading object from OSS")
       temp_filename = os.path.join(local_directory, 'csv_file.csv')
       bucket.get_object_to_file(object_key, temp_filename)
 
       df = pd.read_csv(temp_filename)
 
       parquet_filename = str(f'{local_directory}/{self.artifact_name}')
       logger.info("Exporting pandas dataframe to %s" % parquet_filename)
       df.to_parquet(
           parquet_filename,
           index=False,
           engine='pyarrow',
           compression='gzip')
 
       return parquet_filename
 
 
if __name__ == "__main__":
   # process arguments
   parser = argparse.ArgumentParser(description="Download URL to a local destination")
   parser.add_argument("bucket", type=str, help="Name of the sample to download")
   parser.add_argument("object_path", type=str, help="Name of the sample to download")
   parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
   parser.add_argument("artifact_type", type=str, help="Output artifact type.")
   parser.add_argument(
       "artifact_description", type=str, help="A brief description of this artifact"
   )
   args = parser.parse_args()
 
   # apply arguments to run
   runner = GetOSSDataRunner(
       wandb.init(job_type="download_file"),
       args.artifact_name,
       args.artifact_type,
       args.artifact_description
   )
 
   with tempfile.TemporaryDirectory() as temp_dir:
       LOCAL_FILE = runner.get_oss_data(args.bucket, args.object_path, temp_dir)
       _ = runner.log_artifact(
           args.artifact_name,
           args.artifact_type,
           args.artifact_description,
           LOCAL_FILE
       )
 
   sys.exit(0)
```
 
This looks intimidating but actually simple, we just download our data from a specific OSS bucket with a specific path. We expect that the path should contain only 1 file, and we get that file. To ease our successsive steps, we log this file in `parquet` format, `parquet` is columnar file format that is really optimised to store data for analytics purposes.
 
To access OSS, we need to use an access key and a secret. To securely use this in a development environment, we can put this info inside the `.env` file. Create the file in our root directory, and put our alibaba cloud access key and secret there.
 
```
# .env
OSS_ACCESS_KEY_ID=s3cr3t                        # aliyun access id - put your own
OSS_ACCESS_KEY_SECRET=s3cr3t                    # aliyun access key - put your own
```
 
We also need to put this parquet file somewhere, we can put this in the machine's local file system. However we might encounter problems when we are working with huge data volumes, therefore we are going to make use of weight's and biases (wandb for simplicity). Wandb has many uses, but here we only utilise the artifact store function. Don't forget to get our wandb access key from `https://wandb.ai/authorize` and put the value from there into the `.env` file as well.
 
```
# .env
...
WANDB_API_KEY=s3cr3t                            # wandb.ai api key - put your own
```

All of our component steps will interact with wandb, so we need to build this as a shared utility. Create a new file to hold this shared utility.

```
mkdir nyc_airbnb/utils
touch nyc_airbnb/utils/base_runner
```

Open this new file and copy paste the following code.

```
# ./nyc_airbnb/utils/base_runner
import wandb
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()


class BaseRunner:
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run

    def log_artifact(self,
                     artifact_name: str,
                     artifact_type: str,
                     artifact_description: str,
                     filename: str) -> wandb.Artifact:
        """Log the provided local filename as an artifact in W&B, and add the artifact path
        to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
        Args:
            artifact_name: name for the artifact
            artifact_type:
                type for the artifact (just a string like "raw_data", "clean_data" and so on)
            artifact_description: a brief description of the artifact
            filename: local filename for the artifact
        Returns:
            Wandb artifact object
        """
        # Log to W&B
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=artifact_description,
        )
        artifact.add_file(filename)
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Uploading {artifact_name} to Weights & Biases")

        # We need to call .wait() method to ensure that artifact transport has completed
        # before we exit this method execution
        if wandb.run.mode == 'online':
            artifact.wait()

        return artifact

    def log_model(self,
                  artifact_name: str,
                  artifact_type: str,
                  artifact_description: str,
                  model_dir: str) -> wandb.Artifact:
        """Log the provided local filename as an artifact in W&B, and add the artifact path
        to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
        Args:
            artifact_name: name for the artifact
            artifact_type:
                type for the artifact (just a string like "raw_data", "clean_data" and so on)
            artifact_description: a brief description of the artifact
            model_dir: local path for the model directory
        Returns:
            Wandb artifact object
        """
        # Log to W&B
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=artifact_description,
        )
        artifact.add_dir(model_dir)
        self.wandb_run.log_artifact(artifact)
        # We need to call .wait() method to ensure that artifact transport has completed
        # before we exit this method execution
        if wandb.run.mode == 'online':
            artifact.wait()

        return artifact

    def retrieve_dataset_artifact(self, artifact_name) -> pd.DataFrame:
        """Retrieve wandb artifact as pandas DataFrame, artifact_name should exist in
        the context of current run. This function will only retrieve dataset artifact,
        not model or any other artifact type.
        Args:
            artifact_name: name for the artifact
        Returns:
            DataFrame representation of the artifact
        """
        artifact_local_path = self.wandb_run.use_artifact(artifact_name).file()

        try:
            data = pd.read_parquet(artifact_local_path)
        except FileNotFoundError as err:
            logger.error(f"{artifact_name} is not found")
            raise err

        return data
```
 
Now we are ready to set this up into our main entry point. Let's open the `MLproject` file and put a new entry point.
 
```
# ./MLproject
...
entry_points:
 main:
   ...
   ...
 get_data:
   parameters:
     bucket:
       description: OSS bucket where data is stored
       type: string
     object_path:
       description: OSS object of dataset
       type: string
     artifact_name:
       description: Name for the output artifact
       type: string
     artifact_type:
       description: Type of the output artifact. This will be used to categorize the artifact in the W&B interface
       type: string
     artifact_description:
       description: A brief description of the output artifact
       type: string
   command: "python nyc_airbnb/get_data.py
       {bucket}
       {object_path}
       {artifact_name}
       {artifact_type}
       {artifact_description}"
```
 
Now open `main.py` and let's call our new entry point from the main entry point.
 
```
# main.py
import json
 
import mlflow
import logging
import hydra
import os
import tempfile
 
from dotenv import load_dotenv
from omegaconf import DictConfig
 
load_dotenv()
 
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()
 
_steps = [
   "download"
]
 
 
@hydra.main(config_name='config')
def go(config: DictConfig):
   # Env preparation
   os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
   os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
 
   # Steps to execute
   steps_par = config['main']['steps']
   active_steps = steps_par.split(",") if steps_par != "all" else _steps
 
   with tempfile.TemporaryDirectory() as tmp_dir:
 
       if "download" in active_steps:
           _ = mlflow.run(
               hydra.utils.get_original_cwd(),
               "get_data",
               parameters={
                   "bucket": config["data"]["bucket"],
                   "object_path": f"{config['data']['object']}",
                   "artifact_name": f"{config['data']['raw_data']}",
                   "artifact_type": "raw_data",
                   "artifact_description": "Raw dataset from data store"
               }
           )
 
if __name__ == '__main__':
   go()
```
 
We call `mlflow.run()` if `download` is called. We pass all arguments needed by `get_data` as parameters of `mlflow.run()`. These arguments are retrieved from `config.yaml` that is currently empty, so our next step is to make this config file.
 
```
# ./config.yaml
main:
 project_name: "housing_price"
 experiment_name: "torch_lasso_model"
 steps: all
data:
 bucket: "dana-mle"
 object: "dataset/full_data/"
```
 
The Hydra library will read this configuration, parse it as python Dict, and we consume this Dict within our `go()` function.
 
### Test run
 
Now we have implemented the skeleton with a single component to download our data. To test run it, simply execute this CLI command.
 
```
mlflow run nyc_airbnb
```
 
mlflow will create a new virtual environment, and run the first step `download` to get our data based on the `config.yaml` file. We can see the retrieved data by logging in to the wandb.ai dashboard.

