# Model Card — E-commerce Customer Churn Prediction

## Model Description
XGBoost classifier tuned with Optuna (30 trials) to predict customer churn within 30 days.
Binary target: churned = no order placed in last 30 days.

## Training Data
- Source: Silver layer (customers, orders, clickstream, returns)
- Feature date: 2026-04-07
- Split: 70% train / 15% validation / 15% test
- Class imbalance handled with SMOTE oversampling

## Performance (test set — never seen during tuning)
| Metric    | Baseline (LR) | XGBoost |
|-----------|--------------|---------|
| Accuracy  | 1.0 | 1.0 |
| Precision | 1.0 | 1.0 |
| Recall    | 1.0 | 1.0 |
| F1        | 1.0 | 1.0 |
| AUC-ROC   | 1.0 | 1.0 |

## Hyperparameters (best from Optuna)
{
  "objective": "binary:logistic",
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 0.9210333331861161,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": false,
  "eval_metric": "logloss",
  "feature_types": null,
  "feature_weights": null,
  "gamma": null,
  "grow_policy": null,
  "importance_type": null,
  "interaction_constraints": null,
  "learning_rate": 0.07092662986482734,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 4,
  "max_leaves": null,
  "min_child_weight": 10,
  "missing": NaN,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 209,
  "n_jobs": -1,
  "num_parallel_tree": null,
  "random_state": 42,
  "reg_alpha": null,
  "reg_lambda": null,
  "sampling_method": null,
  "scale_pos_weight": null,
  "subsample": 0.9560808526789939,
  "tree_method": null,
  "validate_parameters": null,
  "verbosity": null,
  "use_label_encoder": false
}

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
Data Engineering Team | Updated: 2026-04-07
