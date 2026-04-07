import glob
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURE_PATH = 'pipelines/gold/features/ecommerce_feature_store'
PRED_PATH = 'pipelines/gold/predictions/ecommerce_predictions'
METRIC_PATH = 'pipelines/gold/metric_store/ecommerce_metrics'
REPORT_PATH = 'ml/training/drift_reports'
MODEL_PATH = 'ml/training/models'

FEATURES = [
    'days_since_last_order', 'order_count_30d', 'order_count_90d',
    'avg_order_value_90d', 'total_spend_lifetime', 'return_rate',
    'promo_usage_rate', 'session_count_7d', 'cart_abandonment_rate',
    'customer_segment_encoded', 'days_since_signup', 'is_high_value'
]
TARGET = 'churned'
MODEL_VERSION = 'xgb_v1'
DRIFT_THRESHOLD = 0.2

os.makedirs(METRIC_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)


def load_all_features():
    files = glob.glob(f"{FEATURE_PATH}/*.parquet")
    if not files:
        raise FileNotFoundError(f"No feature parquet files found under {FEATURE_PATH}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def load_predictions():
    files = glob.glob(f"{PRED_PATH}/*.parquet")
    if not files:
        return None
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def compute_drift(reference_df, current_df):
    """
    reference_df = training window data
    current_df = recent inference window data
    Returns per-feature drift scores dict.
    """
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df[FEATURES], current_data=current_df[FEATURES])

    drift_results = {}
    try:
        report_dict = snapshot.dict()
        for metric in report_dict.get('metrics', []):
            metric_name = metric.get('metric_name', '')
            config = metric.get('config', {})
            value = metric.get('value')
            if metric_name.startswith('ValueDrift('):
                column = config.get('column')
                if column is not None and value is not None:
                    drift_results[column] = round(float(value), 4)
    except Exception as exc:
        print(f"  Warning: could not parse Evidently output, falling back to PSI: {exc}")

    if not drift_results:
        for col in FEATURES:
            ref_vals = reference_df[col].dropna()
            cur_vals = current_df[col].dropna()
            if len(ref_vals) > 10 and len(cur_vals) > 10:
                ref_hist, bin_edges = np.histogram(ref_vals, bins=10)
                cur_hist, _ = np.histogram(cur_vals, bins=bin_edges)
                ref_pct = ref_hist / len(ref_vals) + 1e-6
                cur_pct = cur_hist / len(cur_vals) + 1e-6
                psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
                drift_results[col] = round(float(psi), 4)

    report_path = f"{REPORT_PATH}/drift_report_{date.today()}.html"
    snapshot.save_html(report_path)
    print(f"  Saved Evidently HTML report: {report_path}")
    return drift_results


def compute_model_metrics(pred_df):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if pred_df is None or len(pred_df) == 0:
        return {}

    valid = pred_df.dropna(subset=['actual_value', 'predicted_value', 'confidence_score'])
    if len(valid) == 0:
        return {}

    metrics = {
        'accuracy': round(accuracy_score(valid['actual_value'], valid['predicted_value']), 4),
        'f1': round(f1_score(valid['actual_value'], valid['predicted_value'], zero_division=0), 4),
        'prediction_count': len(valid),
    }

    if valid['actual_value'].nunique() > 1:
        metrics['auc_roc'] = round(roc_auc_score(valid['actual_value'], valid['confidence_score']), 4)
    else:
        metrics['auc_roc'] = float('nan')

    return metrics


def write_metric_store(drift_scores, model_metrics, ref_start, ref_end, cur_start, cur_end):
    rows = []
    ts = datetime.now().isoformat()

    for metric_name, value in model_metrics.items():
        rows.append({
            'model_version': MODEL_VERSION,
            'evaluation_date': date.today().isoformat(),
            'metric_name': metric_name,
            'metric_value': None if pd.isna(value) else float(value),
            'feature_drift_score': None,
            'data_window_start': cur_start,
            'data_window_end': cur_end,
            '_created_ts': ts,
        })

    for feature, score in drift_scores.items():
        rows.append({
            'model_version': MODEL_VERSION,
            'evaluation_date': date.today().isoformat(),
            'metric_name': f'drift_{feature}',
            'metric_value': None,
            'feature_drift_score': float(score),
            'data_window_start': ref_start,
            'data_window_end': ref_end,
            '_created_ts': ts,
        })

    metrics_df = pd.DataFrame(rows)
    fpath = f"{METRIC_PATH}/metrics_{date.today().strftime('%Y%m%d')}.parquet"
    metrics_df.to_parquet(fpath, index=False)
    print(f"  Metric store written: {fpath} ({len(rows)} rows)")
    return metrics_df


def check_drift_alert(drift_scores):
    high_drift_features = {k: v for k, v in drift_scores.items() if v > DRIFT_THRESHOLD}
    drift_share = len(high_drift_features) / len(FEATURES) if FEATURES else 0

    print("\n--- Drift Alert Check ---")
    print(f"  Features with drift > {DRIFT_THRESHOLD}: {len(high_drift_features)}/{len(FEATURES)}")
    for feature, score in sorted(high_drift_features.items(), key=lambda item: -item[1]):
        print(f"    {feature}: {score}")

    if drift_share > 0.3:
        print(f"\n  ALERT: {drift_share:.0%} of features drifted - trigger retraining")
        return True

    print(f"\n  OK: drift within acceptable range ({drift_share:.0%} of features)")
    return False


if __name__ == '__main__':
    print('=== Phase 9: Metric Store + Drift Monitoring ===')

    all_features = load_all_features()
    all_features['feature_date'] = pd.to_datetime(all_features['feature_date'])
    all_features = all_features.sort_values('feature_date')

    split_idx = int(len(all_features) * 0.70)
    reference_df = all_features.iloc[:split_idx]
    current_df = all_features.iloc[split_idx:]

    ref_start = reference_df['feature_date'].min().date().isoformat()
    ref_end = reference_df['feature_date'].max().date().isoformat()
    cur_start = current_df['feature_date'].min().date().isoformat()
    cur_end = current_df['feature_date'].max().date().isoformat()

    print(f"  Reference window: {ref_start} -> {ref_end} ({len(reference_df):,} rows)")
    print(f"  Current window:   {cur_start} -> {cur_end} ({len(current_df):,} rows)")

    print('\n  Computing feature drift...')
    drift_scores = compute_drift(reference_df, current_df)
    print(f"  Drift scores computed for {len(drift_scores)} features")

    pred_df = load_predictions()
    model_metrics = compute_model_metrics(pred_df)
    if model_metrics:
        print(
            f"  Model metrics - AUC: {model_metrics.get('auc_roc', 'N/A')} | "
            f"F1: {model_metrics.get('f1', 'N/A')}"
        )

    write_metric_store(drift_scores, model_metrics, ref_start, ref_end, cur_start, cur_end)
    should_retrain = check_drift_alert(drift_scores)

    import matplotlib.pyplot as plt

    if drift_scores:
        fig, ax = plt.subplots(figsize=(12, 5))
        features = list(drift_scores.keys())
        scores = list(drift_scores.values())
        colors = ['#E24B4A' if score > DRIFT_THRESHOLD else '#1D9E75' for score in scores]
        ax.barh(features, scores, color=colors, edgecolor='none')
        ax.axvline(DRIFT_THRESHOLD, color='#888', linestyle='--', linewidth=1,
                   label=f'Threshold ({DRIFT_THRESHOLD})')
        ax.set_title('Feature drift scores (current vs reference window)')
        ax.set_xlabel('Drift score (PSI / Evidently)')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{REPORT_PATH}/drift_bar_{date.today()}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('  Drift chart saved')

    print(f"\nPhase 9 complete. Metric store: {METRIC_PATH}")
    print(f"Retraining needed: {should_retrain}")
