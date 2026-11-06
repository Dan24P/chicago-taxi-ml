# Chicago Taxi Machine Learning

Three machine-learning workflows using Chicago taxi trip records: predicting trip totals, identifying high-value trips, and forecasting hourly trip counts. The project combines SQL data preparation, scikit-learn pipelines, model comparison, hyperparameter tuning, evaluation, Docker, and automated CI checks.

## Prediction tasks

| Task | Target | Evaluation approach |
| --- | --- | --- |
| Classification | Whether `trip_total >= $50` | Stratified 80/20 train/test split; five-fold stratified cross-validation on training data |
| Regression | Trip total in dollars (`trip_total`) | 80/20 train/test split; five-fold shuffled cross-validation on training data |
| Forecasting | Number of reported trips per hour | Chronological 70/15/15 train/validation/test split after creating lags; expanding-window cross-validation on training data |

The trip-level models use actual trip duration and distance. They describe completed trips; they are not fare estimates based only on information available before pickup.

## Methods

- **Data preparation:** DuckDB queries the trip database. Classification and regression exclude rows with non-positive or missing duration, distance, or trip total.
- **Features:** duration, distance, sine/cosine encoding of hour, weekday, and taxi company. Four feature sets compare numerical features alone with additional categorical features.
- **Preprocessing:** `ColumnTransformer` handles numerical features and one-hot encoding. Preprocessing sits inside each model's scikit-learn `Pipeline` so it is fitted separately within each cross-validation fold.
- **Models:** linear/logistic regression, random forests, and histogram gradient boosting; classification also compares a decision tree.
- **Tuning:** randomized search with 30 candidates and five cross-validation folds. Classification optimizes average precision; regression and forecasting optimize mean absolute error (MAE).
- **Analysis:** classification includes threshold curves, ROC curves, permutation importance, calibration, and a Brier-score comparison against a constant probability estimated from the training set.

Forecasting aggregates records by hour and uses the previous 1–24 observations plus lags of 168, 336, and 504 observations, with optional hour and weekday features. It compares persistence baselines using the previous hour, day, and week. Evaluation uses observed historical counts as lag inputs; it does not implement recursive forecasts for a future period with no new observations.

## Data and initial setup

Source: City of Chicago [Taxi Trips (2013–2023)](https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew), also described in the [official Data.gov catalog](https://catalog.data.gov/dataset/taxi-trips-2013-2023). This project uses the 2023 records.

Raw data and trained models are not included in Git. All commands below run from the repository root. For local execution, use Python 3.12, matching the Docker image and CI configuration. Alternatively, follow the Docker Compose instructions below without installing Python dependencies on your host.

### 1. Install the project

On Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

The `test` extra installs pytest alongside the project's dependencies.

### 2. Prepare the database once

In the data portal, filter `Trip Start Timestamp` to 2023 and export the records as CSV, retaining the original column names. Create a `data/` directory and save the file as `data/Taxi_2023.csv`.

Import the CSV into `data/taxi.db` and apply the SQL transformation:

```bash
python -m taxi_ml.import_data
```

This shared importer creates `raw_trips`, then executes `sql/transform.sql` to create `trips`. The transformation renames columns and cleans duration and trip-total values before converting them to numbers. It accepts numeric or text inputs for these two fields by casting them to text before string cleaning.

The resulting table contains `trip_id`, `trip_start`, `trip_miles`, `company`, `trip_total`, and `trip_seconds`. All three tasks use this database. The pipeline runner does not download or import data automatically. To import through Docker instead, use the command in the Docker section below.

Rerunning the importer keeps an existing `raw_trips` table and recreates `trips` from it. Replacing the CSV alone will not refresh the stored raw data; reloading a different export is currently a manual database-management step.

## Run locally

Start with a sample:

```bash
python run_pipeline.py --task classification --data-mode sample --sample-size 10000
python run_pipeline.py --task regression --data-mode sample --sample-size 10000
python run_pipeline.py --task forecasting
```

Run classification or regression on all eligible trips:

```bash
python run_pipeline.py --task classification --data-mode full
python run_pipeline.py --task regression --data-mode full
```

| Option | Default | Behavior |
| --- | --- | --- |
| `--task` | `classification` | Selects classification, regression, or forecasting |
| `--data-mode` | `sample` | Selects sampled or full data for classification/regression |
| `--sample-size` | `100000` | Number of cleaned trips to sample; unused in full mode |

`DATA_MODE` and `SAMPLE_SIZE` environment variables can override the data defaults; explicit command-line options take precedence. Sampling uses DuckDB reservoir sampling with seed 42. Use enough rows to represent both classes and support five-fold cross-validation.

Forecasting always aggregates all records in `trips`; the sample options do not affect it.

Each run executes **data preparation → model comparison → tuning → final evaluation**. Classification also runs **model analysis**. A failed stage stops the runner. Outputs use fixed paths and may be overwritten by later runs; tuning logs have timestamped filenames.

Sampling reduces the number of trips used for training, but still runs model comparison and tuning. Full mode loads the selected trip-level data into pandas and may exceed available memory. Docker does not remove that memory requirement.

## Run with Docker Compose

Install Docker Engine and Docker Compose, and save the exported CSV as `data/Taxi_2023.csv`, as described above. Create the host directories before running containers:

```bash
mkdir -p data models results
docker compose config --quiet
docker compose build
```

If you have not already prepared the database locally, run the shared importer inside the container:

```bash
docker compose run --rm classification python -m taxi_ml.import_data
```

Here, `classification` supplies the shared image and volume mounts; the explicit Python command replaces its normal training command. The import prepares the database for all three tasks. The image includes the SQL transformation file, and the CSV and generated database stay in the mounted `data/` directory.

The three services share one image, `taxi-ml:local`, and select different tasks:

```bash
docker compose run --rm classification
docker compose run --rm regression
docker compose run --rm forecasting
```

To change the data settings without editing Compose:

```bash
docker compose run --rm -e DATA_MODE=sample -e SAMPLE_SIZE=10000 classification
docker compose run --rm -e DATA_MODE=full regression
```

The `data/`, `models/`, and `results/` directories are bind-mounted into `/app`, so generated files remain on your computer when the container is removed. Rebuild the image after changing Python source, SQL transformations, or dependencies.

The current Compose configuration runs containers as UID/GID `1000:1000`. If your user has different IDs (`id -u` and `id -g`), adjust the shared `user` setting so the container can write to these directories. `MPLCONFIGDIR=/tmp/matplotlib` gives Matplotlib a writable cache directory.

If your Docker installation requires administrator access, prefix the Docker commands with `sudo`.

## Saved results

The checked-in evaluation files currently contain:

| Task | Evaluated model | Saved test metrics | Source |
| --- | --- | --- | --- |
| Classification | Histogram gradient boosting, numerical features + company | ROC AUC **0.9783**, average precision **0.9235**, F1 **0.8728** | [Evaluation JSON](results/classification/final_evaluation.json) |
| Regression | Random forest, numerical features + weekday + company | MAE **$4.39**, R² **0.6861** | [Evaluation JSON](results/regression/final_evaluation.json) |
| Forecasting | Histogram gradient boosting, lags + hour + weekday | MAE **53.26 trips/hour**, R² **0.9730** | [Evaluation JSON](results/forecasting/final_evaluation.json) |

These files are snapshots of previous experiments. They do not record the exact data export, sample configuration, or dependency versions, so the numbers are not a guarantee for a fresh run. Final model and feature choices are currently specified in each task's tuning/evaluation scripts rather than selected automatically from the comparison results.

## Tests and CI

Run the tests locally:

```bash
python -m pytest -q
```

The current tests cover:

- **Shared import:** importing a tiny CSV into `raw_trips`, applying the SQL transformation, and checking the resulting `trips` values.
- **Classification preparation:** sample size, full-data mode, and filtering invalid trips.

Tests use temporary CSV/database files; they do not need the real taxi dataset or modify your local database.

The [GitHub Actions workflow](.github/workflows/ci.yml) runs on pushes and pull requests. It installs the project with test dependencies, runs pytest, builds the Docker image, and starts the container with `python run_pipeline.py --help`.

The container check verifies runner startup and argument parsing. CI does not train models, evaluate the full dataset, publish an image, or deploy a service.

## Repository layout

```text
.github/workflows/ci.yml     Automated tests and Docker checks
src/taxi_ml/
  import_data.py            Shared CSV import and SQL transformation
  common/                   Shared trip features and preprocessing
  classification/           High-trip-total classification and analysis
  regression/               Trip-total prediction
  forecasting/              Hourly trip-count forecasting
sql/transform.sql           Raw-to-modeling table transformation
test/
  test_import_data.py       CSV import and transformation test
  test_prepare_data.py      Classification preparation tests
run_pipeline.py             Pipeline command-line runner
Dockerfile                  Shared training image
compose.yaml                Task services and persistent output mounts
data/                       Local CSV, DuckDB database, prepared Parquet files
models/                     Generated trained pipelines (.joblib)
results/                    Evaluation JSON, tuning logs, and generated plots
```

`data/`, `models/`, PDF files, and routine experiment JSON outputs are ignored by Git. The tracked reports are each task's `final_evaluation.json` and `model_selection.json`, plus classification's `brier_score.json` and `Importance.json`. Timestamped tuning logs and older regression outputs remain local.

## Current limitations and next improvements

- Classification and regression use random splits; a chronological holdout would better assess performance on future trips.
- Forecasting uses row-based shifts without explicitly filling missing hours or resolving daylight-saving transitions. A lag of 24 rows is not guaranteed to represent exactly 24 elapsed hours.
- Tests currently cover shared import and classification preparation. Regression, forecasting, and model behavior need additional coverage.
- Dependency versions and data exports are not pinned. Recording these alongside the sample settings and model results would improve reproducibility.
- Full-data training needs further memory optimization before it can be treated as a supported workflow on small machines.
