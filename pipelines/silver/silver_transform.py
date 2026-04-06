# pipelines/silver/silver_transform.py
import os, re, glob
import pandas as pd
import numpy as np
from datetime import datetime

BRONZE_PATH = 'pipelines/bronze/data'
SILVER_PATH = 'pipelines/silver/data'
DQ_PATH     = 'pipelines/silver/dq_reports'

def read_bronze(table_name):
    """Read all parquet batches from bronze and concat."""
    files = glob.glob(f"{BRONZE_PATH}/{table_name}/*.parquet")
    if not files: raise FileNotFoundError(f"No bronze data for {table_name}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def write_silver(df, table_name):
    os.makedirs(f"{SILVER_PATH}/{table_name}", exist_ok=True)
    fpath = f"{SILVER_PATH}/{table_name}/silver_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(fpath, index=False)
    print(f"  ✓ {table_name}: {len(df)} rows → {fpath}")
    return df

# ── Customers ──────────────────────────────────────────────────
def clean_customers():
    df = read_bronze('customers')
    bronze_count = len(df)

    # 1. Dedup by customer_id (keep first occurrence)
    df = df.drop_duplicates(subset=['customer_id'], keep='first')

    # 2. Title-case full_name
    df['full_name'] = df['full_name'].str.title()

    # 3. Parse mixed signup_date formats
    def parse_date(v):
        if pd.isna(v): return None
        v = str(v).strip()
        for fmt in ('%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y'):
            try: return pd.to_datetime(v, format=fmt).date().isoformat()
            except: pass
        return None
    df['signup_date'] = df['signup_date'].apply(parse_date)

    # 4. Fill null city → 'Unknown'
    df['city'] = df['city'].fillna('Unknown')

    # 5. SHA-256 hash email (PII masking)
    df['email_hashed'] = df['email'].apply(
        lambda e: __import__('hashlib').sha256(str(e).encode()).hexdigest() if pd.notna(e) else None)
    df = df.drop(columns=['email'])

    write_silver(df, 'customers')
    return df, bronze_count

# ── Orders ─────────────────────────────────────────────────────
def clean_orders():
    df = read_bronze('orders')
    bronze_count = len(df)

    # Normalise 6 status variants → 3 canonical states
    STATUS_MAP = {
        'placed':'PLACED','Placed':'PLACED','order_placed':'PLACED',
        'new':'PLACED','PLACED':'PLACED','pending':'PLACED',
        'shipped':'SHIPPED','Shipped':'SHIPPED','dispatched':'SHIPPED',
        'in_transit':'SHIPPED','SHIPPED':'SHIPPED','out_for_delivery':'SHIPPED',
        'cancelled':'CANCELLED','Canceled':'CANCELLED','CANCELLED':'CANCELLED',
        'cancel':'CANCELLED','void':'CANCELLED','VOID':'CANCELLED'
    }
    df['status'] = df['status'].map(STATUS_MAP).fillna('UNKNOWN')

    # Strip commas from total_amount, cast to float
    df['total_amount'] = (df['total_amount'].astype(str)
                          .str.replace(',', '', regex=False)
                          .str.strip())
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

    # Flag null ship_date rows
    df['has_missing_ship_date'] = df['ship_date'].isna()

    write_silver(df, 'orders')
    return df, bronze_count

# ── Order Items ────────────────────────────────────────────────
def clean_order_items():
    df = read_bronze('order_items')
    bronze_count = len(df)

    # Ensure quantity and unit_price are numeric
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0.0)

    # Flag invalid quantities (negative or zero)
    df['has_invalid_quantity'] = df['quantity'] <= 0

    write_silver(df, 'order_items')
    return df, bronze_count

# ── Products ───────────────────────────────────────────────────
def clean_products():
    df = read_bronze('products')
    bronze_count = len(df)

    # Strip HTML tags from description
    def strip_html(text):
        if pd.isna(text): return text
        return re.sub(r'<[^>]+>', '', str(text)).strip()
    df['description'] = df['description'].apply(strip_html)

    # Strip trailing whitespace from product_name
    df['product_name'] = df['product_name'].str.strip()

    # Fill null category_id → sentinel -1
    df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(-1).astype(int)

    write_silver(df, 'products')
    return df, bronze_count

# ── Inventory ──────────────────────────────────────────────────
def clean_inventory():
    df = read_bronze('inventory')
    bronze_count = len(df)

    # Strip trailing space from warehouse_id
    df['warehouse_id'] = df['warehouse_id'].str.strip()

    # Clip negative stock_qty to 0, flag as write-off
    df['stock_qty']   = pd.to_numeric(df['stock_qty'], errors='coerce').fillna(0)
    df['is_writeoff'] = df['stock_qty'] < 0
    df['stock_qty']   = df['stock_qty'].clip(lower=0)

    write_silver(df, 'inventory')
    return df, bronze_count

# ── Returns ────────────────────────────────────────────────────
def clean_returns():
    df = read_bronze('returns')
    bronze_count = len(df)

    # Strip 'Rs.' prefix and cast to float
    df['refund_amount'] = (df['refund_amount'].astype(str)
                           .str.replace('Rs.', '', regex=False)
                           .str.strip())
    df['refund_amount'] = pd.to_numeric(df['refund_amount'], errors='coerce')

    # Flag missing order_id
    df['has_missing_order_id'] = df['order_id'].isna()

    write_silver(df, 'returns')
    return df, bronze_count

# ── Clickstream ────────────────────────────────────────────────
def clean_clickstream():
    df = read_bronze('clickstream_events')
    bronze_count = len(df)

    # Fix event_type typos
    TYPO_FIX = {'veiw': 'view', 'clck': 'click'}
    df['event_type'] = df['event_type'].replace(TYPO_FIX)

    # Unify timestamps: Unix epoch → ISO datetime
    def unify_ts(v):
        if pd.isna(v): return None
        v = str(v).strip()
        try:  # Unix timestamp (numeric string)
            ts = float(v)
            if ts > 1e9:  # looks like epoch
                return datetime.fromtimestamp(ts).isoformat()
        except ValueError:
            pass
        try:  # Already ISO
            return pd.to_datetime(v).isoformat()
        except:
            return None
    df['event_timestamp'] = df['timestamp'].apply(unify_ts)
    df = df.drop(columns=['timestamp'])

    # Flag null session_id (keep the rows)
    df['has_null_session'] = df['session_id'].isna()

    write_silver(df, 'clickstream_events')
    return df, bronze_count

# ── Data Quality Gate ──────────────────────────────────────────
def run_dq_gate(table_name, silver_df, bronze_count, pk_col):
    os.makedirs(DQ_PATH, exist_ok=True)
    issues = []
    passed = True

    # Check 1: PK null rate must be 0%
    pk_nulls = silver_df[pk_col].isna().sum()
    if pk_nulls > 0:
        issues.append(f"CRITICAL: {pk_col} has {pk_nulls} nulls")
        passed = False

    # Check 2: Row count >= 80% of bronze
    ratio = len(silver_df) / bronze_count if bronze_count > 0 else 0
    if ratio < 0.80:
        issues.append(f"CRITICAL: row count {len(silver_df)} is only {ratio:.1%} of bronze {bronze_count}")
        passed = False

    report = {
        'table': table_name, 'timestamp': datetime.now().isoformat(),
        'bronze_count': bronze_count, 'silver_count': len(silver_df),
        'row_ratio': round(ratio, 4), 'pk_nulls': int(pk_nulls),
        'passed': passed, 'issues': issues
    }
    import json
    rpath = f"{DQ_PATH}/{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rpath, 'w') as f:
        json.dump(report, f, indent=2)

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  DQ {status} — {table_name}: {len(silver_df)}/{bronze_count} rows, {pk_nulls} PK nulls")
    if not passed:
        raise ValueError(f"DQ gate FAILED for {table_name}: {issues}")
    return report

if __name__ == '__main__':
    print("=== Phase 3: Silver Transformation ===")
    TABLES = [
        ('customers',  clean_customers,  'customer_id'),
        ('orders',     clean_orders,     'order_id'),
        ('order_items', clean_order_items, 'order_item_id'),
        ('products',   clean_products,   'product_id'),
        ('inventory',  clean_inventory,  'inventory_id'),
        ('returns',    clean_returns,    'return_id'),
        ('clickstream_events', clean_clickstream, 'event_id'),
    ]
    for name, fn, pk in TABLES:
        try:
            df, bronze_count = fn()
            run_dq_gate(name, df, bronze_count, pk)
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    print("\n✅ Silver transformation complete! Check pipelines/silver/data/")