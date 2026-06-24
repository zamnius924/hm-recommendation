# H&M Personalized Fashion Recommendations

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
├── requirements.txt
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── schemas.py              # Pydantic models
│   └── services/
│       ├── get_con.py          # DuckDB connection
│       ├── get_customers.py
│       ├── get_recommendations.py
│       └── get_recommendations_batch.py
│
├── data/
│   ├── raw/                    # original .parquet files
│   │   ├── transactions_train.parquet
│   │   ├── articles.parquet
│   │   └── customers.parquet
│   ├── processed/              # intermediate & final datasets
│   └── recommendations/        # pre‑computed recommendations for API
│
├── models/                     # saved models & params
│   ├── als_best_params.json
│   ├── ltr_best_params.json
│   ├── ltr_model.cbm
│   └── ltr_model_info.json
│
└── scripts/                    # reusable modules
    ├── ap_at_k.py
    ├── build_mapping.py
    ├── generate_als_candidates.py
    ├── generate_features.py
    ├── generate_pool.py
    ├── generate_scores.py
    ├── load_db.py
    ├── map_at_k.py
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

### 5. Start the API server

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

## 📦 Dependencies

- Python ≥ 3.9

- `numpy`, `pandas`, `scipy`

- `duckdb`, `pyarrow`

- `implicit` (ALS)

- `catboost` (LTR)

- `optuna` (hyperparameter optimisation)

- `fastapi`, `uvicorn` (API serving)

Full list in `requirements.txt`.

## 🙏 Acknowledgements

This project uses the __H&M Personalized Fashion Recommendations__ dataset from Kaggle. The original data is provided by H&M Group.

## 📄 License

MIT License. Feel free to use and modify for your own projects.