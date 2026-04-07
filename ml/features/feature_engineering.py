import glob
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

SILVER = 'pipelines/silver/data'
FEATURE_PATH = 'pipelines/gold/features/ecommerce_feature_store'
FEATURE_DATE = date.today()


def read_silver(table):
    files = glob.glob(f"{SILVER}/{table}/*.parquet")
    if not files:
        raise FileNotFoundError(f"No parquet files found for silver table: {table}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def compute_features(feature_date=FEATURE_DATE):
    """
    Point-in-time safe: all features use only data up to feature_date.
    Entity = customer_id.
    """
    print(f"Computing features for date: {feature_date}")
    cutoff = pd.Timestamp(feature_date)

    customers = read_silver('customers')
    orders = read_silver('orders')
    read_silver('order_items')
    returns = read_silver('returns')
    clicks = read_silver('clickstream_events')

    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    orders = orders[orders['order_date'] <= cutoff].copy()

    clicks['event_timestamp'] = pd.to_datetime(clicks['event_timestamp'], errors='coerce')
    clicks = clicks[clicks['event_timestamp'] <= cutoff].copy()

    returns['return_date'] = pd.to_datetime(returns['return_date'], errors='coerce')
    returns = returns[returns['return_date'] <= cutoff].copy()

    feat = customers[['customer_id', 'segment', 'signup_date']].copy()
    feat['signup_date'] = pd.to_datetime(feat['signup_date'], errors='coerce')
    feat['days_since_signup'] = (cutoff - feat['signup_date']).dt.days.fillna(0)

    last_order = orders.groupby('customer_id')['order_date'].max().reset_index()
    last_order.columns = ['customer_id', 'last_order_date']
    feat = feat.merge(last_order, on='customer_id', how='left')
    feat['days_since_last_order'] = (cutoff - feat['last_order_date']).dt.days.fillna(999)

    for window_days, col_name in [(30, 'order_count_30d'), (90, 'order_count_90d')]:
        window_start = cutoff - timedelta(days=window_days)
        window_orders = orders[orders['order_date'] >= window_start]
        counts = window_orders.groupby('customer_id')['order_id'].count().reset_index()
        counts.columns = ['customer_id', col_name]
        feat = feat.merge(counts, on='customer_id', how='left')
        feat[col_name] = feat[col_name].fillna(0)

    window_90 = orders[orders['order_date'] >= cutoff - timedelta(days=90)]
    aov_90 = window_90.groupby('customer_id')['total_amount'].mean().reset_index()
    aov_90.columns = ['customer_id', 'avg_order_value_90d']
    feat = feat.merge(aov_90, on='customer_id', how='left')
    feat['avg_order_value_90d'] = feat['avg_order_value_90d'].fillna(0)

    ltv = orders.groupby('customer_id')['total_amount'].sum().reset_index()
    ltv.columns = ['customer_id', 'total_spend_lifetime']
    feat = feat.merge(ltv, on='customer_id', how='left')
    feat['total_spend_lifetime'] = feat['total_spend_lifetime'].fillna(0)
    feat['is_high_value'] = (feat['total_spend_lifetime'] > 5000).astype(int)

    total_orders_per_cust = orders.groupby('customer_id')['order_id'].count().reset_index()
    total_orders_per_cust.columns = ['customer_id', 'total_orders']

    valid_returns = returns[returns['order_id'].notna()].copy()
    if 'customer_id' not in valid_returns.columns:
        order_cust = orders[['order_id', 'customer_id']].drop_duplicates()
        ret_with_cust = valid_returns.merge(order_cust, on='order_id', how='left')
    else:
        ret_with_cust = valid_returns.copy()
        if ret_with_cust['customer_id'].isna().any():
            order_cust = orders[['order_id', 'customer_id']].drop_duplicates()
            missing_customer = ret_with_cust['customer_id'].isna()
            ret_with_cust.loc[missing_customer, 'customer_id'] = (
                ret_with_cust.loc[missing_customer, ['order_id']]
                .merge(order_cust, on='order_id', how='left')['customer_id']
                .values
            )

    ret_with_cust = ret_with_cust[ret_with_cust['customer_id'].notna()].copy()
    ret_counts = ret_with_cust.groupby('customer_id')['return_id'].count().reset_index()
    ret_counts.columns = ['customer_id', 'return_count']

    feat = feat.merge(total_orders_per_cust, on='customer_id', how='left')
    feat = feat.merge(ret_counts, on='customer_id', how='left')
    feat['total_orders'] = feat['total_orders'].fillna(0)
    feat['return_count'] = feat['return_count'].fillna(0)
    feat['return_rate'] = np.where(
        feat['total_orders'] > 0,
        feat['return_count'] / feat['total_orders'],
        0,
    )

    promo_orders = orders[orders['promo_id'].notna()]
    promo_counts = promo_orders.groupby('customer_id')['order_id'].count().reset_index()
    promo_counts.columns = ['customer_id', 'promo_order_count']
    feat = feat.merge(promo_counts, on='customer_id', how='left')
    feat['promo_order_count'] = feat['promo_order_count'].fillna(0)
    feat['promo_usage_rate'] = np.where(
        feat['total_orders'] > 0,
        feat['promo_order_count'] / feat['total_orders'],
        0,
    )

    click_7d = clicks[clicks['event_timestamp'] >= cutoff - timedelta(days=7)]
    session_cnt = click_7d.groupby('customer_id')['session_id'].nunique().reset_index()
    session_cnt.columns = ['customer_id', 'session_count_7d']
    feat = feat.merge(session_cnt, on='customer_id', how='left')
    feat['session_count_7d'] = feat['session_count_7d'].fillna(0)

    cart_clicks = clicks[clicks['event_type'].isin(['add_to_cart', 'purchase'])]
    cart_summary = cart_clicks.groupby('customer_id')['event_type'].value_counts().unstack(fill_value=0)
    if 'add_to_cart' in cart_summary.columns and 'purchase' in cart_summary.columns:
        cart_summary['cart_abandonment_rate'] = (
            (cart_summary['add_to_cart'] - cart_summary['purchase'].clip(upper=cart_summary['add_to_cart']))
            / cart_summary['add_to_cart'].replace(0, np.nan)
        ).fillna(0)
        feat = feat.merge(
            cart_summary[['cart_abandonment_rate']].reset_index(),
            on='customer_id',
            how='left',
        )
    else:
        feat['cart_abandonment_rate'] = 0
    feat['cart_abandonment_rate'] = feat['cart_abandonment_rate'].fillna(0)

    segment_map = {'premium': 2, 'regular': 1, 'occasional': 0}
    feat['customer_segment_encoded'] = feat['segment'].map(segment_map).fillna(1)
    feat['churned'] = (feat['days_since_last_order'] > 30).astype(int)

    feature_cols = [
        'customer_id',
        'days_since_last_order',
        'order_count_30d',
        'order_count_90d',
        'avg_order_value_90d',
        'total_spend_lifetime',
        'return_rate',
        'promo_usage_rate',
        'session_count_7d',
        'cart_abandonment_rate',
        'customer_segment_encoded',
        'days_since_signup',
        'is_high_value',
        'churned',
    ]

    feat_final = feat[feature_cols].copy()
    feat_final['feature_date'] = pd.Timestamp(feature_date).date().isoformat()
    feat_final['_created_ts'] = datetime.now().isoformat()

    feat_final['days_since_last_order'] = feat_final['days_since_last_order'].clip(upper=365)
    feat_final['return_rate'] = feat_final['return_rate'].clip(0, 1)
    feat_final['cart_abandonment_rate'] = feat_final['cart_abandonment_rate'].clip(0, 1)

    print(f"  Feature shape: {feat_final.shape}")
    print(f"  Churn rate: {feat_final['churned'].mean():.1%}")
    print(f"  Null check:\n{feat_final.isnull().sum()[feat_final.isnull().sum() > 0]}")

    return feat_final


def save_features(feat_df, feature_date=FEATURE_DATE):
    os.makedirs(FEATURE_PATH, exist_ok=True)
    fpath = f"{FEATURE_PATH}/features_{pd.Timestamp(feature_date).strftime('%Y%m%d')}.parquet"
    feat_df.to_parquet(fpath, index=False)
    print(f"\nFeature store saved: {fpath}")
    print(f"   Rows: {len(feat_df):,} | Columns: {feat_df.shape[1]}")


if __name__ == '__main__':
    print('=== Phase 7: Feature Engineering ===')
    features = compute_features()
    save_features(features)
    print('\n--- Feature preview ---')
    print(features.describe().round(2))
