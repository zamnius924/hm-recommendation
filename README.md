# H&M Personalized Fashion Recommendations

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Airflow](https://img.shields.io/badge/Airflow-3.2-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)
![CatBoost](https://img.shields.io/badge/CatBoost-LTR-orange)
![Implicit ALS](https://img.shields.io/badge/Implicit-ALS-lightgrey)
![Optuna](https://img.shields.io/badge/Optuna-HPO-purple)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-yellow)

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

## 🗂️ Project Structure

```
│
├── 01_loader.py                # create temporal split dates
├── 02_als_calibration.py       # tune ALS hyperparameters
├── 03_feature_engineering.py   # generate candidates + features for all splits
├── 04_ltr_calibration.py       # tune CatBoost hyperparameters
├── 05_ltr_fit.py               # train final model & evaluate
├── 06_recommendations.py       # generate final recommendations for API
│
├── dags/                        # Airflow DAGs
│   ├── calibrate_dag.py         # hyperparameter calibration
│   └── recommend_dag.py         # daily retraining & inference
│
├── pipeline/                   # production‑ready entry points (used by Airflow)
│   ├── run_update_windows.py
│   ├── run_als_calibration.py
│   ├── run_feature_engineering.py
│   ├── run_ltr_calibration.py
│   ├── run_ltr_fit.py
│   └── run_recommendations.py
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
│   ├── processed/              # intermediate datasets (local development)
│   ├── production/             # datasets used for model retraining (Airflow)
│   └── recommendations/        # pre‑computed recommendations for API
│
├── models/
│   ├── config/                 # hyperparameter configs (dev)
│   │   ├── als_best_params.json
│   │   ├── ltr_best_params.json
│   │   └── window_config.json
│   └── production/             # production models & configs
│       ├── config/
│       │   ├── als_best_params.json
│       │   ├── ltr_best_params.json
│       │   └── window_config.json
│       ├── ltr_model.cbm
│       └── ltr_model_info.json
│
└── scripts/                    # reusable modules
    ├── __init__.py
    ├── ap_at_k.py
    ├── build_mapping.py
    ├── generate_als_candidates.py
    ├── generate_features.py
    ├── generate_pool.py
    ├── generate_scores.py
    ├── load_db.py
    ├── load_window_config.py
    ├── map_at_k.py
    ├── paths.py
    ├── sparse_interaction_matrix.py
    ├── tuning_objective_als.py
    ├── tuning_objective_ltr.py
    └── window_extraction.py
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

Execute the scripts in order:

```bash
python 01_loader.py              # split dates
python 02_als_calibration.py     # tune ALS
python 03_feature_engineering.py # generate candidates & features
python 04_ltr_calibration.py     # tune LTR
python 05_ltr_fit.py             # train final model
python 06_recommendations.py     # create recommendations for API
```

All models and parameters will be saved in the `models/` folder, and the final recommendations table will be stored as `data/recommendations/df_rec.parquet`.

### 5. Start Airflow (optional)

The project includes two production Airflow DAGs for automated retraining.

Run Airflow in project-local mode:
```bash
export AIRFLOW_HOME=$(pwd)/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
airflow standalone
```
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

- **Development** (`models/config/`, `data/processed/`): Local experimentation and one‑time runs  
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