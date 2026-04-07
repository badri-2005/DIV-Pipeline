# ml/training/train_churn.py
import os, glob, json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

FEATURE_PATH  = 'pipelines/gold/features/ecommerce_feature_store'
PRED_PATH     = 'pipelines/gold/predictions/ecommerce_predictions'
MODEL_PATH    = 'ml/training/models'
MLFLOW_URI    = 'ml/training/mlruns'

FEATURES = [
    'days_since_last_order', 'order_count_30d', 'order_count_90d',
    'avg_order_value_90d', 'total_spend_lifetime', 'return_rate',
    'promo_usage_rate', 'session_count_7d', 'cart_abandonment_rate',
    'customer_segment_encoded', 'days_since_signup', 'is_high_value'
]
TARGET = 'churned'

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PRED_PATH, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("ecommerce_churn_prediction")

# ── Load feature store ─────────────────────────────────────────
def load_features():
    files = glob.glob(f"{FEATURE_PATH}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Loaded features: {df.shape} | Churn rate: {df[TARGET].mean():.1%}")
    return df

# ── Train/val/test split ───────────────────────────────────────
def split_data(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    cids = df['customer_id'].values

    X_temp, X_test, y_temp, y_test, cids_temp, cids_test = train_test_split(
        X, y, cids, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val, cids_train, cids_val = train_test_split(
        X_temp, y_temp, cids_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, cids_test

# ── Metrics helper ─────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc':   round(roc_auc_score(y_true, y_prob), 4),
    }

# ── Model 1: Logistic Regression baseline ─────────────────────
def train_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n--- Training baseline: Logistic Regression ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_tr_s, y_train)

        val_pred  = lr.predict(X_va_s)
        val_prob  = lr.predict_proba(X_va_s)[:, 1]
        val_metrics = get_metrics(y_val, val_pred, val_prob)

        test_pred = lr.predict(X_te_s)
        test_prob = lr.predict_proba(X_te_s)[:, 1]
        test_metrics = get_metrics(y_test, test_pred, test_prob)

        mlflow.log_params({'model': 'logistic_regression', 'class_weight': 'balanced'})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(lr, "logistic_regression")

    print(f"  Baseline val  AUC: {val_metrics['auc_roc']} | F1: {val_metrics['f1']}")
    print(f"  Baseline test AUC: {test_metrics['auc_roc']} | F1: {test_metrics['f1']}")
    return test_metrics, test_pred, test_prob

# ── Model 2: XGBoost + Optuna + SMOTE ─────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n--- Training XGBoost with Optuna tuning ---")

    # Handle class imbalance with SMOTE on training set only
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE — train size: {len(X_res):,} | churn rate: {y_res.mean():.1%}")

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
            'max_depth':       trial.suggest_int('max_depth', 3, 8),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
            'use_label_encoder': False, 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1
        }
        model = XGBClassifier(**params)
        model.fit(X_res, y_res, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    print(f"  Best val F1: {study.best_value:.4f}")
    print(f"  Best params: {best_params}")

    with mlflow.start_run(run_name="xgboost_optuna"):
        best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss',
                            'random_state': 42, 'n_jobs': -1})
        xgb = XGBClassifier(**best_params)
        xgb.fit(X_res, y_res)

        val_pred  = xgb.predict(X_val)
        val_prob  = xgb.predict_proba(X_val)[:, 1]
        val_metrics = get_metrics(y_val, val_pred, val_prob)

        test_pred = xgb.predict(X_test)
        test_prob = xgb.predict_proba(X_test)[:, 1]
        test_metrics = get_metrics(y_test, test_pred, test_prob)

        mlflow.log_params({'model': 'xgboost_optuna', **best_params})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.xgboost.log_model(xgb, "xgboost_model")

        # SHAP values
        explainer   = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test[:500])

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:500], feature_names=FEATURES,
                          show=False, plot_type='bar')
        plt.title('SHAP feature importance (XGBoost churn model)')
        plt.tight_layout()
        shap_path = f'{MODEL_PATH}/shap_bar.png'
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(shap_path)
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not churned','Churned'],
                    yticklabels=['Not churned','Churned'])
        ax.set_title('Confusion matrix (test set)')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        cm_path = f'{MODEL_PATH}/confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save model locally
        import pickle
        with open(f'{MODEL_PATH}/xgb_churn.pkl', 'wb') as f:
            pickle.dump(xgb, f)

    print(f"  XGBoost test AUC: {test_metrics['auc_roc']} | F1: {test_metrics['f1']}")
    return xgb, test_metrics, test_pred, test_prob

# ── Write predictions to Gold layer ───────────────────────────
def save_predictions(df, test_pred, test_prob, cids_test, model_version='xgb_v1'):
    from datetime import date
    pred_rows = []
    for cid, pred, prob, actual in zip(cids_test, test_pred, test_prob,
                                        df.set_index('customer_id').loc[cids_test, TARGET]):
        pred_rows.append({
            'customer_id':       cid,
            'prediction_date':   date.today().isoformat(),
            'predicted_value':   int(pred),
            'confidence_score':  round(float(prob), 4),
            'model_version':     model_version,
            'actual_value':      int(actual)
        })
    pred_df = pd.DataFrame(pred_rows)
    os.makedirs(PRED_PATH, exist_ok=True)
    fpath = f"{PRED_PATH}/predictions_{date.today().strftime('%Y%m%d')}.parquet"
    pred_df.to_parquet(fpath, index=False)
    print(f"\n✅ Predictions saved: {fpath}")
    return pred_df

# ── Model card ─────────────────────────────────────────────────
def write_model_card(baseline_metrics, xgb_metrics, best_params):
    from datetime import date
    card = f"""# Model Card — E-commerce Customer Churn Prediction

## Model Description
XGBoost classifier tuned with Optuna (30 trials) to predict customer churn within 30 days.
Binary target: churned = no order placed in last 30 days.

## Training Data
- Source: Silver layer (customers, orders, clickstream, returns)
- Feature date: {date.today()}
- Split: 70% train / 15% validation / 15% test
- Class imbalance handled with SMOTE oversampling

## Performance (test set — never seen during tuning)
| Metric    | Baseline (LR) | XGBoost |
|-----------|--------------|---------|
| Accuracy  | {baseline_metrics['accuracy']} | {xgb_metrics['accuracy']} |
| Precision | {baseline_metrics['precision']} | {xgb_metrics['precision']} |
| Recall    | {baseline_metrics['recall']} | {xgb_metrics['recall']} |
| F1        | {baseline_metrics['f1']} | {xgb_metrics['f1']} |
| AUC-ROC   | {baseline_metrics['auc_roc']} | {xgb_metrics['auc_roc']} |

## Hyperparameters (best from Optuna)
{json.dumps(best_params, indent=2)}

## Intended Use
Internal customer retention team. Identify at-risk customers for proactive outreach.

## Limitations
- Does not capture seasonal patterns (LSTM planned for v2)
- Churn definition (30-day inactivity) may miss intentional dormant users
- Training data period: last 1 year only

## Known Biases
- 'Occasional' segment may be over-predicted as churned due to inherently low purchase frequency
- New customers (<30 days since signup) excluded from meaningful churn scoring

## Owner
Data Engineering Team | Updated: {date.today()}
"""
    card_path = f'{MODEL_PATH}/model_card.md'
    with open(card_path, 'w') as f:
        f.write(card)
    print(f"  ✓ Model card: {card_path}")
    return card_path

# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Phase 8: ML Model Training ===")
    df = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test, cids_test = split_data(df)

    # Baseline
    baseline_metrics, bl_pred, bl_prob = train_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # XGBoost main model
    xgb_model, xgb_metrics, xgb_pred, xgb_prob = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # Save predictions and model card
    save_predictions(df, xgb_pred, xgb_prob, cids_test)
    write_model_card(baseline_metrics, xgb_metrics, xgb_model.get_params())

    print("\n=== Final Results ===")
    print(f"Baseline AUC: {baseline_metrics['auc_roc']} | XGBoost AUC: {xgb_metrics['auc_roc']}")
    print(f"XGBoost beats baseline: {xgb_metrics['auc_roc'] > baseline_metrics['auc_roc']}")
    print(f"\nRun 'mlflow ui --backend-store-uri ml/training/mlruns' to view experiment tracking")