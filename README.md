# Steam Pre-launch Success Predictor

An automated ML pipeline that predicts whether a Steam game will reach at least 100 reviews with at least 80% positive reviews. The model uses only information available before release.

## Requirements

- Python 3.12
- Docker with Docker Compose
- GNU Make
- `cron` for scheduled runs on Linux or macOS

## Run the project

Create the virtual environment and install dependencies:

```bash
make setup
```

Download the dataset, prepare train/test data, and train the model:

```bash
make download
make pipeline
```

Build and start the API and web application:

```bash
make deploy
```

Open:

- Web application: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- API health check: <http://localhost:8000/health>

Stop the containers:

```bash
make clean
```

## Pipeline

The pipeline has three stages:

1. **Data engineering:** download, clean, remove duplicates and price outliers, create the target, and save stratified train/test splits.
2. **Model engineering:** build text and numeric features, train and evaluate LightGBM, log test metrics to MLflow, and save the packaged model.
3. **Deployment:** build and run separate FastAPI and Streamlit containers.

DVC manages the first two stages in `dvc.yaml`:

```text
data/raw/steam_games.csv
          |
       prepare
          |
 train.csv + test.csv
          |
        train
          |
 model.joblib + metrics.json
```

Run the DVC pipeline directly with:

```bash
.venv/bin/dvc repro
```

DVC reruns only stages whose code, data, dependencies, or parameters have changed.

## Automated run every five minutes

Run the complete pipeline once:

```bash
./scripts/run_pipeline.sh
```

This command downloads the data if necessary, runs `dvc repro`, and rebuilds and starts the Docker services.

Install the cron schedule:

```bash
./scripts/install_cron.sh
crontab -l
```

Scheduled output is written to `logs/pipeline.log`. The installer replaces an existing cron entry for this project. It requires a Unix-like environment; on Windows, run it through WSL or configure Task Scheduler separately.

## Metrics and MLflow

Latest test metrics are saved to `models/metrics.json` and logged to the local MLflow store in `mlruns/`.

Display metrics with DVC:

```bash
.venv/bin/dvc metrics show
```

Start the MLflow UI:

```bash
.venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open <http://localhost:5000>.

## API example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "genre": "Action;Adventure;Indie",
    "tags": "Co-op;Multiplayer;Atmospheric",
    "categories": "Online Co-op;Steam Achievements",
    "developer": "Example Studio",
    "publisher": "Example Publisher",
    "price": 19.99,
    "platforms": ["windows", "linux"],
    "languages": ["English", "Russian"],
    "required_age": 12
  }'
```

## Configuration

Data paths, target thresholds, and model hyperparameters are defined in `params.yaml`. After changing them, run:

```bash
make pipeline
```

## Repository structure

```text
code/
  datasets/          # dataset download and preparation
  models/            # feature engineering, training, evaluation, packaging
  deployment/
    api/              # FastAPI service and Dockerfile
    app/              # Streamlit application and Dockerfile
    docker-compose.yml
data/raw/             # downloaded source data
data/processed/       # DVC train/test outputs
models/               # packaged model and test metrics
scripts/              # complete pipeline and cron installer
dvc.yaml              # DVC pipeline definition
dvc.lock              # recorded dependency and output hashes
params.yaml           # data and model configuration
requirements.txt
```

## Dataset

[All 55,000 Games on Steam (November 2022)](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022), licensed under CC BY-SA 4.0.
