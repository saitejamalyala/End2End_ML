# End to End machine learning workflow [![Python application](https://github.com/saitejamalyala/End2End_ML/actions/workflows/python-train_pred.yml/badge.svg)](https://github.com/saitejamalyala/End2End_ML/actions/workflows/python-train_pred.yml)
Used Car Prediction
==============================

## Instructions
This repo contains sample code for the End to End machine learning for used car price prediction.
Data set used for the project is [Vehicle dataset from kaggle](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)

- Suggested to try the project by creating a virtual environment
  ```
  python -m venv env_end2endML
  ```
- Activate the virutal environment
- Install the dependencies using the command
  ```
  pip install -r requirements.txt 
  ```

## To train the model :
1. [Download the data](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)
2. Modify the configuration according to needs in `config.yaml`
3. Train the model by running `python src/train.py`
---
## Experiments and artifacts(Code, data, model) tracking using weights and biases
4. After step 3, you will be prompted to enter wandb login details to track the experiments, metrics and artifcats
5. For hyper parameter sweep run  ``` wandb agent username/project-name/sweepcode ```
6. Check previous experiments and sweeps [here](https://wandb.ai/saitejam/stepstone-demo/sweeps/z605w0e2)
7. Check model artifacts with versioning [here](https://wandb.ai/saitejam/stepstone-demo/artifacts/models/trained_model/1c93794f899d85543a6f/files) 
---
## CI/CD using github actions, unit testing with pytest
#### This repo contains two workflows
8. [Python applicaton work flow](https://github.com/saitejamalyala/End2End_ML/blob/main/.github/workflows/python-train_pred.yml) to check if training pipeline is not broken after every push/pull. 
9. Extend test cases if needed [here](https://github.com/saitejamalyala/End2End_ML/blob/main/tests/test_create_dataset.py) or [here](https://github.com/saitejamalyala/End2End_ML/blob/main/tests/test_create_feat_dataset.py)
10. [Docker build workflow](https://github.com/saitejamalyala/End2End_ML/blob/main/.github/workflows/python_test_stapp.yml) to check if docker image building is successful after every push/pull.
---
## Deploying the model using an webapp
11. Make predictions on the validation set by running `python src/predict.py`
12. To run the streamlit app `streamlit run src/app.py`
13. After step 12 open the [browser](http://localhost:8501/) if you are running in your local machine or check step 14
14. Check the [public app](https://share.streamlit.io/saitejamalyala/end2end_ml/main/app/stepstoneapp.py)

## Serving model using FAST API
15. To serve using fast API using following commands 
  ``` 
  cd fastapi 
  ```
  ```
  uvicorn main:app -reload
  ```
16. [check on local host](http://127.0.0.1:8000/docs)

## Building Docker Image
16. Configure the Docker image by editing the `Dockerfile` 
17. Build the docker image using  `sudo docker run -p 8051:8051 appname:tag`

### Followed by container orchestration and model monitoring
This benchmark took approximately 0.5 hours to execute on a Windows 10 laptop with 12GB of RAM and 8 cores at 2GHz.


