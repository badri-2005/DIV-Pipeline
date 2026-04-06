# pipelines/gold/gold_kpis.py
import os, glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SILVER_PATH = 'pipelines/silver/data'
GOLD_PATH   = 'pipelines/gold/data'

def read_silver(table):
    files = glob.glob(f"{SILVER_PATH}/{table}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def write_gold(df, kpi_name):
    os.makedirs(f"{GOLD_PATH}/{kpi_name}", exist_ok=True)
    fpath = f"{GOLD_PATH}/{kpi_name}/gold_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(fpath, index=False)
    print(f"  ✓ KPI saved: {kpi_name} ({len(df)} rows)")

def kpi_gmv():
    """KPI 1: Gross Merchandise Value — daily trend"""
    orders = read_silver('orders')
    items  = read_silver('order_items')
    active = orders[~orders['status'].isin(['CANCELLED', 'VOID'])][['order_id','order_date']]
    merged = active.merge(items[['order_id','quantity','unit_price']], on='order_id')
    merged['revenue'] = pd.to_numeric(merged['quantity'], errors='coerce') * \
                        pd.to_numeric(merged['unit_price'], errors='coerce')
    gmv = merged.groupby('order_date')['revenue'].sum().reset_index()
    gmv.columns = ['order_date', 'gmv']
    write_gold(gmv, 'kpi_01_gmv')
    return gmv

def kpi_aov():
    """KPI 8: Average Order Value by segment"""
    orders   = read_silver('orders')
    customers = read_silver('customers')
    active = orders[~orders['status'].isin(['CANCELLED','VOID'])]
    merged = active.merge(customers[['customer_id','segment']], on='customer_id', how='left')
    aov = merged.groupby('segment').agg(
        aov=('total_amount','mean'),
        order_count=('order_id','nunique')
    ).reset_index()
    aov['aov'] = aov['aov'].round(2)
    write_gold(aov, 'kpi_08_aov')
    return aov

def kpi_repeat_purchase_rate():
    """KPI 9: Repeat purchase rate"""
    orders = read_silver('orders')
    counts = orders.groupby('customer_id')['order_id'].count().reset_index()
    counts.columns = ['customer_id','order_count']
    counts['is_repeat'] = counts['order_count'] >= 2
    rate = counts['is_repeat'].mean()
    result = pd.DataFrame([{'metric':'repeat_purchase_rate', 'value': round(rate, 4)}])
    write_gold(result, 'kpi_09_repeat_purchase_rate')
    return result

def kpi_return_rate_by_category():
    """KPI 6: Return rate by category"""
    orders   = read_silver('orders')
    items    = read_silver('order_items')
    products = read_silver('products')
    returns  = read_silver('returns')

    order_items_products = items.merge(products[['product_id','category_name']], on='product_id', how='left')
    total_by_cat = order_items_products.groupby('category_name')['order_id'].nunique().reset_index()
    total_by_cat.columns = ['category_name', 'total_orders']

    valid_returns = returns[returns['order_id'].notna()].merge(
        order_items_products[['order_id','category_name']].drop_duplicates(), on='order_id', how='left')
    returns_by_cat = valid_returns.groupby('category_name')['return_id'].nunique().reset_index()
    returns_by_cat.columns = ['category_name', 'return_count']

    result = total_by_cat.merge(returns_by_cat, on='category_name', how='left')
    result['return_count'] = result['return_count'].fillna(0)
    result['return_rate']  = (result['return_count'] / result['total_orders']).round(4)
    result['high_risk']    = result['return_rate'] > 0.15
    write_gold(result, 'kpi_06_return_rate_by_category')
    return result

def kpi_inventory_turnover():
    """KPI 7: Inventory turnover (simplified without COGS — use revenue as proxy)"""
    inventory = read_silver('inventory')
    products  = read_silver('products')
    merged = inventory.merge(products[['product_id','price']], on='product_id', how='left')
    merged['inventory_value'] = merged['stock_qty'] * pd.to_numeric(merged['price'], errors='coerce')
    by_product = merged.groupby('product_id').agg(
        avg_inventory_value=('inventory_value','mean')
    ).reset_index()
    by_product['is_dead_stock'] = by_product['avg_inventory_value'] > 0  # simplified flag
    write_gold(by_product, 'kpi_07_inventory_turnover')
    return by_product

if __name__ == '__main__':
    print("=== Phase 4: Gold KPI Layer ===")
    kpi_gmv()
    kpi_aov()
    kpi_repeat_purchase_rate()
    kpi_return_rate_by_category()
    kpi_inventory_turnover()
    print("\n✅ Gold KPIs complete! Check pipelines/gold/data/")