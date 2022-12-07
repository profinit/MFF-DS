# DataScience Technologies II (MLOps)

Quickstart guide:

- Follow steps `1, 2, 3` to run MLFlow. Then open another terminal and follow steps `1.5, 2, 4` to run `jupyter lab` or use your environment in your IDE.

## 1. Install fresh virtualenv

This exercise was prepared with Python 3.9.10, please have it (or `python>=3.9`) installed. If you have different version, it might work but you need to figure out the correct dependencies yourself.

see https://docs.python.org/3/library/venv.html

If you are using **VSCode** or **PyCharm**, find a guide on how to create your environment there.

### Conda

```
conda create --name mlops python=3.9.10
```

If you have any issues, refer to e.g., https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf or elsewhere.

### Unix

```bash
python -m venv ./venv_mlops 
```

### Windows -- powershell

```ps
python -m venv ./venv_mlops 
```

## 1.5 Activate your environment

### Conda

```
conda activate mlops
```

### Unix

```bash
source ./venv_mlops/bin/activate
```

### Windows -- powershell

```ps
./venv_mlops/Scripts/Activate.ps1
```


## 2. Install dependencies

```
pip install -r requirements.txt
```


## 3. Run MLFlow

Simple as this:

```bash
mlflow ui
```

### (OPTIONAL) Install SQLite 3

if you want to play with registering Models

see https://www.sqlite.org/download.html

If you are on Linux:

```bash
sudo apt-get install sqlite3
sqlite3 ./analysis/mlflow.db ".databases"
```

Then run

```bash
cd analysis
mlflow server --backend-store-uri "sqlite:///$PWD/mlflow.db" --default-artifact-root file:$PWD/mlruns -h localhost -p 5000
```

## 4. Run your jupyter server

```bash
jupyter lab
```


## MLFlow cheatsheet

This needs to be set before any MLFlow API calls, if you have MLFlow on localhost. If you have your MLFlow elsewhere, edit accordingly.

```bash
export MLFLOW_TRACKING_URI=localhost:5000
```

or you can set it programmatically:

```python
mlflow.set_tracking_uri("localhost:5000")
```

### Experiment name

```python
mlflow.set_experiment("my-experiment")
```

### Logging a run

```python
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
```

### Parameters

Hyperparameters, reference to training data, model type

```python
mlflow.log_param("learning_rate", 0.01)
```

### Metrics

```python
train_auc = .75
mlflow.log_metric("train_auc", train_auc)
```

### Tags

```python
mlflow.set_tags({"A": "B", "C": "D"})
```

### Additional Artifacts

```python
mlflow.log_artifact("fancy-graph-name.png")
```

### Loading a model 

If you use database storage along with MLFlow for your registered models

```python
mlflow.pyfunc.load_model('models:/titanic/2')
```

### Autolog

```python
mlflow.sklearn.autolog()
model = LogisticRegression()
with mlflow.start_run() as run:
    model.fit(X, y)
```

