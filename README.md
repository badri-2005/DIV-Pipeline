# Data Pipeline Project

This project builds an end-to-end e-commerce data platform from synthetic source generation to analytics dashboards and ML monitoring.

It covers:
- raw data generation from databases and files
- bronze ingestion
- silver cleaning and data quality checks
- gold KPI creation
- feature engineering for churn modeling
- model training and prediction generation
- drift monitoring and metric store creation
- dashboard generation with DuckDB and Plotly

## Project Flow

The currently implemented runnable phases in this repo are:

1. Phase 1: Data Generation
2. Phase 2: Bronze Ingestion
3. Phase 3: Silver Transformation
4. Phase 4: Gold KPI Layer
5. Phase 7: Feature Engineering
6. Phase 8: ML Model Training
7. Phase 9: Metric Store + Drift Monitoring
8. Dashboard Build and Visualization

Note:
- I did not find separate runnable scripts labeled Phase 5 or Phase 6 in the current codebase.
- Dashboard generation is implemented as a separate analytics step after the gold and ML layers.

## Folder Structure

```text
data_generation/          Synthetic source generation
pipelines/bronze/         Raw ingestion into bronze parquet
pipelines/silver/         Cleaning, standardization, DQ checks
pipelines/gold/           KPI outputs, features, metrics, predictions
ml/features/              Feature engineering
ml/training/              Training, monitoring, model artifacts
dashboards/               DuckDB build + Plotly dashboard
```

## Environment Setup

### 1. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

If `requirements.txt` is not filled yet, install the packages used by the project manually.

Typical packages used here:

```powershell
pip install pandas numpy pyarrow fastparquet faker sqlalchemy pymysql psycopg2-binary python-dotenv pillow openpyxl scikit-learn imbalanced-learn xgboost mlflow shap optuna matplotlib seaborn duckdb plotly evidently
```

### 3. Configure environment variables

Create or update `.env` with your database connection details:

```env
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_HOST=...
MYSQL_PORT=3306
MYSQL_DB=...

POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_HOST=...
POSTGRES_PORT=5432
POSTGRES_DB=...
```

## Overall Run Commands: Phase 1 To Last Phase

Run these from the project root:

```powershell
.\venv\Scripts\Activate.ps1
python data_generation\generate_data.py
python pipelines\bronze\bronze_ingest.py
python pipelines\silver\silver_transform.py
python pipelines\gold\gold_kpis.py
python ml\features\feature_engineering.py
python ml\training\train_churn.py
python ml\training\drift_monitor.py
python dashboards\quick_dashboard.py
```

If you want to build only the DuckDB dashboard database first:

```powershell
python dashboards\create_duckdb.py
```

If you want to inspect MLFlow locally:

```powershell
mlflow ui --backend-store-uri ml/training/mlruns
```

## Output Artifacts By Phase

### Phase 1: Data Generation

Generates source data into:
- MySQL tables: `customers`, `orders`, `order_items`
- PostgreSQL tables: `products`, `inventory`
- files under `data_generation/raw_files/`

Main outputs:
- `clickstream_events.json`
- `returns.json`
- `promotions.csv`
- `product_catalogue.csv`
- `weekly_sales_summary.xlsx`
- review `.txt` files
- product image `.jpg` files

### Phase 2: Bronze Ingestion

Reads all source systems and writes raw parquet batches into:
- `pipelines/bronze/data/customers/`
- `pipelines/bronze/data/orders/`
- `pipelines/bronze/data/order_items/`
- `pipelines/bronze/data/products/`
- `pipelines/bronze/data/inventory/`
- `pipelines/bronze/data/returns/`
- `pipelines/bronze/data/clickstream_events/`
- and other bronze folders

Each batch includes ingestion metadata such as:
- `_source_system`
- `_ingest_ts`
- `_ingest_date`
- `_batch_id`
- `_file_name`

### Phase 3: Silver Transformation

Cleans and standardizes bronze data into silver parquet tables.

Key transformations:
- normalize customer names
- parse mixed date formats
- hash customer emails
- standardize order statuses
- cast numeric fields
- fix clickstream typos
- unify timestamps
- flag data quality issues

Outputs:
- `pipelines/silver/data/...`
- DQ reports in `pipelines/silver/dq_reports/`

### Phase 4: Gold KPI Layer

Builds business KPIs from silver data.

Implemented KPIs:
- GMV
- AOV
- repeat purchase rate
- return rate by category
- inventory turnover

Outputs:
- `pipelines/gold/data/kpi_01_gmv/`
- `pipelines/gold/data/kpi_06_return_rate_by_category/`
- `pipelines/gold/data/kpi_07_inventory_turnover/`
- `pipelines/gold/data/kpi_08_aov/`
- `pipelines/gold/data/kpi_09_repeat_purchase_rate/`

### Phase 7: Feature Engineering

Builds a customer-level feature store for churn modeling.

Key features:
- recency
- 30-day and 90-day order counts
- average order value
- lifetime spend
- return rate
- promo usage rate
- session count
- cart abandonment rate
- encoded segment
- signup age
- high-value flag
- churn label

Output:
- `pipelines/gold/features/ecommerce_feature_store/`

### Phase 8: ML Model Training

Trains churn models using the feature store.

Models used:
- Logistic Regression baseline
- XGBoost main model

Supporting components:
- SMOTE for imbalance handling
- Optuna for tuning
- MLFlow for experiment tracking
- SHAP for feature importance

Outputs:
- predictions in `pipelines/gold/predictions/ecommerce_predictions/`
- model files in `ml/training/models/`
- MLFlow runs in `ml/training/mlruns/`

### Phase 9: Drift Monitoring

Monitors feature drift and writes monitoring metrics.

What it does:
- compares reference and current feature windows
- computes drift with Evidently or PSI fallback
- writes drift metrics to metric store
- saves drift HTML report and chart
- triggers retraining alert logic

Outputs:
- `pipelines/gold/metric_store/ecommerce_metrics/`
- `ml/training/drift_reports/`

### Dashboard Step

Builds a DuckDB analytics database from silver and gold layers, then renders an HTML dashboard.

Outputs:
- `dashboards/ecommerce_dashboard.duckdb`
- `dashboards/dashboard.html`

## Complete Study Docs: Step-By-Step Process

### Step 1: Understand the architecture

This project follows a layered data engineering design:
- source systems and raw files
- bronze raw landing
- silver cleaned and standardized layer
- gold business and ML-ready outputs
- serving layer for dashboards and monitoring

This pattern is important because each layer has a clear responsibility.

### Step 2: Learn the source systems

The project uses both structured and semi-structured sources:
- MySQL for transactional customer/order data
- PostgreSQL for products and inventory
- CSV for product and promotion feeds
- JSON for clickstream and returns
- Excel for weekly sales summaries
- TXT reviews
- image metadata sources

Study goal:
- understand how real-world pipelines rarely use one perfect source
- see how noisy data is intentionally introduced for realism

### Step 3: Study data generation

In `data_generation/generate_data.py`, the project creates synthetic data with deliberate data quality problems.

Examples of injected noise:
- mixed-case names
- duplicate emails
- null values
- mixed date formats
- malformed numeric values
- inconsistent status labels
- typo event names
- missing keys
- corrupted images

Why this matters:
- it simulates messy production data
- it gives the downstream pipeline something meaningful to clean

### Step 4: Study bronze ingestion

In `pipelines/bronze/bronze_ingest.py`, the project ingests all source systems with minimal transformation.

Bronze principles used here:
- preserve raw data as much as possible
- attach ingestion metadata
- write immutable-style batch parquet files

Why bronze exists:
- reprocessing becomes easier
- lineage is clearer
- audits and troubleshooting are simpler

### Step 5: Study silver transformation

In `pipelines/silver/silver_transform.py`, the project standardizes and validates data.

Important concepts:
- schema cleanup
- type casting
- standardization
- null handling
- deduplication
- business-rule cleanup
- data quality gates

This is where raw data becomes reliable enough for analytics and ML.

### Step 6: Study data quality gates

Each silver table is checked with rules such as:
- primary key null count
- row retention ratio from bronze to silver

Why this matters:
- prevents silently bad downstream outputs
- makes failures explicit
- creates operational trust in the pipeline

### Step 7: Study the gold layer

In `pipelines/gold/gold_kpis.py`, the project builds business-facing KPI outputs.

This is the first layer that business users would directly consume.

Main KPI ideas:
- GMV tells total sales trend
- AOV shows purchasing behavior
- repeat purchase rate measures loyalty
- return rate highlights category issues
- inventory turnover helps operations

### Step 8: Study feature engineering

In `ml/features/feature_engineering.py`, customer-level features are built for churn prediction.

Concepts to focus on:
- point-in-time correctness
- entity-level feature tables
- behavioral aggregation windows
- target leakage avoidance

Important feature categories:
- recency
- frequency
- monetary value
- returns and dissatisfaction
- promotions
- engagement
- churn target

Why this is important:
- good ML performance depends more on feature quality than model choice alone

### Step 9: Study model training

In `ml/training/train_churn.py`, the project trains two models.

Pipeline learning points:
- baseline model first
- train/validation/test split
- imbalance correction with SMOTE
- hyperparameter tuning with Optuna
- experiment tracking with MLFlow
- explainability with SHAP

Why this matters:
- it reflects a practical ML workflow, not just a single `.fit()` call

### Step 10: Study prediction outputs

After training, predictions are saved with:
- customer ID
- prediction date
- predicted label
- confidence score
- model version
- actual value

This mirrors real prediction stores used for reporting and monitoring.

### Step 11: Study drift monitoring

In `ml/training/drift_monitor.py`, the project compares historical and current feature distributions.

Core monitoring ideas:
- feature drift detection
- reference vs current windows
- monitoring metric store
- retraining triggers

Why this matters:
- models degrade over time
- monitoring is required for production ML reliability

### Step 12: Study dashboard serving

The dashboard layer uses:
- DuckDB as a lightweight analytics engine
- Plotly for visualization
- HTML output for easy sharing

This is useful because it shows how parquet-based data products can be turned into a quick analytics interface without needing a full BI platform.

### Step 13: Understand the end-to-end value chain

The full learning sequence is:
- create raw data
- ingest it
- clean it
- validate it
- aggregate it
- engineer features
- train models
- monitor drift
- visualize outputs

This is the core story of modern data engineering plus applied ML operations.

## Recommended Study Order

Follow this order when presenting or revising the project:

1. Read `data_generation/generate_data.py`
2. Read `pipelines/bronze/bronze_ingest.py`
3. Read `pipelines/silver/silver_transform.py`
4. Read `pipelines/gold/gold_kpis.py`
5. Read `ml/features/feature_engineering.py`
6. Read `ml/training/train_churn.py`
7. Read `ml/training/drift_monitor.py`
8. Read `dashboards/create_duckdb.py`
9. Read `dashboards/quick_dashboard.py`

## Quick Demo Order

If you need to demo the project quickly:

1. Run all commands from top to bottom.
2. Open `dashboards/dashboard.html`.
3. Show KPI parquet outputs in `pipelines/gold/data/`.
4. Show feature store output in `pipelines/gold/features/`.
5. Show prediction output in `pipelines/gold/predictions/`.
6. Show model artifacts in `ml/training/models/`.
7. Show drift reports in `ml/training/drift_reports/`.
8. Optionally launch MLFlow UI.

## Important Notes

- Run the scripts from the repo root: `D:\Data-Pipeline`
- Activate the same `venv` before running commands
- MySQL and PostgreSQL must be reachable for phases 1 and 2
- If dashboard DB is missing, `quick_dashboard.py` can rebuild it
- The dashboard uses `ecommerce_dashboard.duckdb`
- The monitoring script depends on the installed Evidently version in the current `venv`

## Future Improvements

- fill and lock `requirements.txt`
- add a single orchestration script for all phases
- add explicit Phase 5 and Phase 6 modules if they are part of the intended design
- add tests for feature engineering and monitoring
- document database bootstrap steps for MySQL and PostgreSQL
