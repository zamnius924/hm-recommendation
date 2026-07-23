# H&M Personalized Fashion Recommendations

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![Implicit ALS](https://img.shields.io/badge/Implicit-ALS-lightgrey)
![CatBoost](https://img.shields.io/badge/CatBoost-LTR-F5A623)
![Optuna](https://img.shields.io/badge/Optuna-HPO-8A2BE2)
![Airflow](https://img.shields.io/badge/Airflow-3.2-017CEE?logo=apacheairflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?logo=fastapi&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-FFF000?logo=duckdb&logoColor=black)

Production-ready recommendation system combining collaborative filtering, learning-to-rank, Airflow orchestration, and FastAPI serving.

A two‑stage recommendation system that combines **Matrix Factorization (ALS)** with a **Learning‑to‑Rank (LTR)** model (CatBoost) to provide personalised fashion recommendations for H&M customers.

The pipeline is built around a sliding‑window temporal split and produces a ranked list of up to 100 articles per customer, served via a **FastAPI** web service.

## 📌 Key Features

- **Collaborative Filtering** with implicit feedback (Alternating Least Squares)  
- **Learning‑to‑Rank** with CatBoost (YetiRank loss) to re‑rank ALS candidates  
- Rich feature engineering: user, article, and user‑article interaction features  
- Temporal cross‑validation (train / validation / test) with fixed 8‑week feature windows and 6‑day target windows  
- Hyperparameter tuning with **Optuna** (for both ALS and LTR)  
- REST API (FastAPI) serving recommendations with article metadata  
- In‑memory DuckDB for fast SQL feature engineering
- **Automated retraining pipeline** with Apache Airflow

## 🧠 Architecture Overview

1. **ALS** generates a set of candidate articles (top‑K) for each customer based on implicit feedback.  
2. **Feature engineering** adds user, article, and cross‑features to each (customer, article) candidate pair.  
3. **CatBoost Ranker** (LTR) learns to re‑rank the candidates using the engineered features and a binary target (purchased in the next week).  
4. The final model is served through a lightweight **FastAPI** application that loads pre‑computed recommendations from Parquet.

The project hosts **two retrieval models sharing the LTR reranking stage**, organised as `pipelines/model_1` (ALS) and `pipelines/model_2` (Two-Tower neural encoder). The steps above describe model 1; model 2 swaps the ALS candidate generator for a two-tower model and reuses the same LTR stage. Model 1 is the complete, served pipeline; model 2's LTR stages are still in progress.

## 🗂️ Project Structure

```
│
├── pipelines/                   # entry scripts (# %% cells), split per model
│   ├── 01_loader.py             # shared: temporal split dates
│   ├── model_1/                 # ALS → LTR
│   │   ├── 02_als_calibration.py
│   │   ├── 03_feature_engineering.py
│   │   ├── 04_ltr_calibration.py
│   │   ├── 05_ltr_fit.py
│   │   └── 06_recommendations.py
│   └── model_2/                 # Two-Tower → LTR (WIP: LTR stages pending)
│       ├── 02_tt_train.py
│       └── 03_feature_engineering.py
│
├── orchestration/               # Airflow orchestration (dags + production runners)
│   ├── dags/                    # Airflow DAGs
│   │   ├── calibrate_dag.py     # hyperparameter calibration
│   │   └── recommend_dag.py     # daily retraining & inference
│   └── pipeline/                # production‑ready entry points (called by the DAGs)
│       ├── run_update_windows.py
│       ├── run_als_calibration.py
│       ├── run_feature_engineering.py
│       ├── run_ltr_calibration.py
│       ├── run_ltr_fit.py
│       └── run_recommendations.py
│
├── requirements.txt
├── README.md
│
├── app/                        # FastAPI service
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── get_con.py
│       ├── get_customers.py
│       ├── get_recommendations.py
│       └── get_recommendations_batch.py
│
├── data/
│   ├── raw/                    # original .parquet files
│   │   ├── transactions_train.parquet
│   │   ├── articles.parquet
│   │   └── customers.parquet
│   ├── processed/              # intermediate datasets (dev)
│   │   ├── model_1/            # ALS candidates + LTR features
│   │   └── model_2/            # Two-Tower candidates + LTR features
│   ├── production/             # datasets used for retraining (Airflow)
│   └── recommendations/        # pre‑computed recommendations for API
│
├── models/
│   ├── config/                 # shared config (window_config.json)
│   ├── model_1/                # ALS → LTR artifacts
│   │   ├── config/             # als_best_params.json, ltr_best_params.json
│   │   ├── ltr_model.cbm
│   │   └── ltr_model_info.json
│   ├── model_2/                # Two-Tower artifacts
│   │   ├── config/
│   │   └── tt_model.pt
│   └── production/             # production models & configs (Airflow)
│       ├── config/
│       └── ltr_model.cbm
│
└── scripts/                    # reusable library modules (model-agnostic)
    ├── als/                    # ALS candidates, sparse matrix, tuning
    ├── two_tower/              # towers, datasets, candidates, loss
    ├── ltr/                    # pool, scores, LTR tuning
    ├── features/               # customer / article / cross features
    ├── data/                   # DB load/save, windows, mapping
    ├── evaluation/             # map@k, ap@k
    └── utils/                  # paths, dtypes, logger
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hm-recommendation.git
cd hm-recommendation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The original H&M dataset is available on [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview) or via [Hugging Face](https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data).

Place the `.parquet` files inside `data/raw/`:

1) `transactions_train.parquet`

2) `articles.parquet`

3) `customers.parquet`

### 4. Run the training pipeline

Both model pipelines share the loader (`01_loader.py`) and the LTR reranking stage. Run the `# %%` scripts in order, from the repository root.

**Shared**
```bash
python pipelines/01_loader.py                      # temporal split dates
```

**Model 1 — ALS → LTR**
```bash
python pipelines/model_1/02_als_calibration.py     # tune ALS
python pipelines/model_1/03_feature_engineering.py # ALS candidates + features
python pipelines/model_1/04_ltr_calibration.py     # tune LTR
python pipelines/model_1/05_ltr_fit.py             # train final ranker
python pipelines/model_1/06_recommendations.py     # recommendations for API
```

**Model 2 — Two-Tower → LTR** (retrieval stages; LTR stages WIP)
```bash
python pipelines/model_2/02_tt_train.py            # train two-tower encoder
python pipelines/model_2/03_feature_engineering.py # TT candidates + features
```

Per-model artifacts are saved under `models/model_1/` and `models/model_2/`, while the shared `window_config.json` stays in `models/config/`. Model 1's recommendations are written to `data/recommendations/model_1/df_rec.parquet` — the table the API serves.

### 5. Start Airflow (optional)

The project includes two production Airflow DAGs for automated retraining.

Run Airflow in project-local mode (execute from the repository root):
```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/orchestration/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```
> The `export`s must come **before** `airflow standalone`: a process reads its
> environment at startup, so Airflow picks up `AIRFLOW_HOME` (where to create/read
> `.airflow`) and the `AIRFLOW__CORE__*` overrides only if they are already set.
> These `AIRFLOW__SECTION__KEY` variables override settings at runtime and are **not**
> written to `airflow.cfg` — the file keeps its defaults, but the effective value is
> the env one (env > `airflow.cfg` > built-in defaults). Verify with
> `airflow config get-value core dags_folder`.

Then open:
```
http://localhost:8080
```
The generated admin password will be printed during the first launch or stored in
```
.airflow/simple_auth_manager_passwords.json.generated
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/customers?start=1&end=10` | GET | List customer IDs by row number |
| `/recommendations/{customer_id}?k=12` | GET | Top‑K recommendations for a single customer |
| `/recommendations_batch` | POST | Batch recommendations for multiple customers |

## Example

__Request__

```bash
http://127.0.0.1:8000/recommendations/00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657?k=1
```

__Response__

```json
{
  "customer_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657",
  "recommendations": [
    {
      "rating": 1,
      "article_id": "0568601006",
      "product_type_name": "Blazer",
      "graphical_appearance_name": "Solid",
      "detail_desc": "Fitted jacket in woven fabric with notch lapels, jetted front pockets, a decorative button at the cuffs and a single back vent. Lined."
    }
  ]
}
```

## 🧪 Evaluation

Model performance is measured using __Mean Average Precision at K (MAP@K)__ with `K = 12`.

| Model | MAP@12 (test) |
|-------|---------------|
| Most Popular | ~0.015 |
| ALS | ~0.027 |
| ALS + CatBoost | **0.0336** |

The final model achieves a MAP@12 of __0.0336__ on the held‑out test period.

## ⚙️ Temporal Split Strategy

- __Feature window__: 8 weeks of historical transaction data

- __Target window__: 7 days immediately following the feature window

- __Train/Validation/Test__ are shifted back in time so that no data leakage occurs between splits

| Split | Feature Window | Target Window |
|----------|--------|-------------|
| Train | weeks 3 – 10 (`2020-07-21` – `2020-09-15`) | week 11 (`2020-09-16` – `2020-09-22`) |
| Validation | weeks 2 – 9 (`2020-07-14` – `2020-09-08`) | week 10 (`2020-09-09` – `2020-09-15`)|
| Test | weeks 1 – 8 (`2020-07-07` – `2020-09-01`) | week 9 (`2020-09-02` – `2020-09-08`) |

During model training only customers and articles that appear in the feature window are considered.

## 🧬 Feature Engineering

A rich set of features is built for each (customer, article) candidate:

- __Customer features__: number of purchases, unique articles, purchase days, recency, average price, etc.

- __Article features__: popularity, recency of purchases, average customer age, price, sales channel mix, etc.

- __Customer‑article features__: prior purchase indicators, days since last purchase of the same article/type/colour/garment group, purchase counts, price deviation, etc.

- __ALS score__: the raw collaborative filtering score for the candidate pair.

All feature engineering is performed using __DuckDB__ for speed and scalability.

## 🔄 Automation with Airflow

The production pipeline is fully automated using **Apache Airflow**, with two separate DAGs for calibration and daily retraining.

### Calibration DAG (`calibrate_dag.py`)

Runs on a **weekly** schedule to re‑tune hyperparameters as new data arrives:

> update_windows → calibrate_als → feature_engineering → calibrate_ltr

| Task | Description |
|------|-------------|
| `update_windows` | Recalculates train/validation/test date windows based on the latest data |
| `calibrate_als` | Tunes ALS hyperparameters using Optuna on the validation set |
| `feature_engineering` | Generates ALS candidates + features for train and validation sets |
| `calibrate_ltr` | Tunes CatBoost hyperparameters using Optuna |

### Recommendation DAG (`recommend_dag.py`)

Runs on a **daily** schedule to retrain the model and generate fresh recommendations:

> update_windows → feature_engineering → fit_ltr → generate_recommendations


| Task | Description |
|------|-------------|
| `update_windows` | Recalculates train/test date windows based on the latest data |
| `feature_engineering` | Generates ALS candidates + features for train and test sets (production mode) |
| `fit_ltr` | Trains the final CatBoost model and validates quality (fails if MAP@12 < 0.03) |
| `generate_recommendations` | Produces recommendations and saves them for the API |

### Production vs Development Separation

- **Development** (`pipelines/`, `models/model_1|2/`, `data/processed/model_1|2/`): Local experimentation and one‑time runs  
- **Production** (`models/production/`, `data/production/`): Airflow‑managed retraining with separate configs and datasets

This ensures that the automated pipeline doesn't interfere with ongoing experimentation.

## 📦 Dependencies

- Python ≥ 3.9

- `numpy`, `pandas`, `scipy`

- `duckdb`, `pyarrow`

- `implicit` (ALS)

- `catboost` (LTR)

- `optuna` (hyperparameter optimisation)

- `fastapi`, `uvicorn` (API serving)

- `apache-airflow` (orchestration)

Full list in `requirements.txt`.

## 🙏 Acknowledgements

This project uses the __H&M Personalized Fashion Recommendations__ dataset from Kaggle. The original data is provided by H&M Group.

## 📄 License

MIT License. Feel free to use and modify for your own projects.