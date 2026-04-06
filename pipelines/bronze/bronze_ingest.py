# pipelines/bronze/bronze_ingest.py
import os, json, glob
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from io import StringIO

load_dotenv()

BRONZE_PATH = 'pipelines/bronze/data'
BATCH_ID = datetime.now().strftime('%Y%m%d_%H%M%S')


# ── Common Utility ─────────────────────────────────────────────
def add_bronze_meta(df, source_system, file_name=None):
    df = df.copy()
    df['_source_system'] = source_system
    df['_ingest_ts']     = datetime.now().isoformat()
    df['_ingest_date']   = datetime.now().date().isoformat()
    df['_batch_id']      = BATCH_ID
    if file_name:
        df['_file_name'] = file_name
    return df


def append_bronze(df, table_name):
    path = f"{BRONZE_PATH}/{table_name}"
    os.makedirs(path, exist_ok=True)
    fname = f"{path}/batch_{BATCH_ID}.parquet"
    df.to_parquet(fname, index=False)
    print(f"  ✓ {table_name}: {len(df)} rows → {fname}")


# ── 1. MySQL ──────────────────────────────────────────────────
def ingest_mysql():
    url = (f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
           f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}")
    
    engine = create_engine(url)

    for table in ['customers', 'orders', 'order_items']:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        df = add_bronze_meta(df, f'mysql_{table}')
        append_bronze(df, table)


# ── 2. PostgreSQL ─────────────────────────────────────────────
def ingest_postgres():
    url = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
           f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}")
    
    engine = create_engine(url)

    for table in ['products', 'inventory']:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        df = add_bronze_meta(df, f'postgres_{table}')
        append_bronze(df, table)


# ── 3. CSV: product_catalogue (FIXED) ─────────────────────────
def ingest_product_catalogue():
    path = 'data_generation/raw_files/product_catalogue.csv'

    # Step 1: Normalize delimiters (; → ,)
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = [line.replace(';', ',') for line in f if line.strip()]

    # Step 2: Read safely
    df = pd.read_csv(
        StringIO(''.join(lines)),
        header=None,
        names=['product_id','product_name','price','stock','supplier'],
        on_bad_lines='skip'
    )

    print(f"[INFO] product_catalogue rows loaded: {len(df)}")

    df = add_bronze_meta(df, 'file_product_catalogue', file_name=path)
    append_bronze(df, 'product_catalogue')


# ── 4. CSV: promotions ────────────────────────────────────────
def ingest_promotions():
    df = pd.read_csv('data_generation/raw_files/promotions.csv', dtype=str)
    df = add_bronze_meta(df, 'file_promotions', file_name='promotions.csv')
    append_bronze(df, 'promotions')


# ── 5. JSON: clickstream ─────────────────────────────────────
def ingest_clickstream():
    path = 'data_generation/raw_files/clickstream_events.json'
    
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # ✅ FIX: Normalize timestamp column
    def normalize_timestamp(x):
        if pd.isna(x):
            return None
        try:
            # If integer → convert from Unix
            if isinstance(x, (int, float)):
                return pd.to_datetime(x, unit='s')
            # If string → parse normally
            return pd.to_datetime(x)
        except:
            return None

    df['timestamp'] = df['timestamp'].apply(normalize_timestamp)

    # Optional: convert to string for parquet safety
    df['timestamp'] = df['timestamp'].astype(str)

    print(f"[INFO] clickstream rows: {len(df)}")

    df = add_bronze_meta(df, 'file_clickstream', file_name=path)
    append_bronze(df, 'clickstream_events')


# ── 6. JSON: returns ─────────────────────────────────────────
def ingest_returns():
    with open('data_generation/raw_files/returns.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Clean refund_amount column: remove 'Rs.' prefix and convert to float
    if 'refund_amount' in df.columns:
        df['refund_amount'] = df['refund_amount'].astype(str).str.replace('Rs.', '', regex=False).astype(float)

    df = add_bronze_meta(df, 'file_returns', file_name='returns.json')
    append_bronze(df, 'returns')


# ── 7. Excel ─────────────────────────────────────────────────
def ingest_weekly_sales():
    path = 'data_generation/raw_files/weekly_sales_summary.xlsx'
    xl = pd.ExcelFile(path)

    all_sheets = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str)
        df['_sheet_name'] = sheet
        all_sheets.append(df)

    combined = pd.concat(all_sheets, ignore_index=True)
    combined = add_bronze_meta(combined, 'file_weekly_sales_excel', file_name=path)
    append_bronze(combined, 'weekly_sales_summary')


# ── 8. TXT reviews ───────────────────────────────────────────
def ingest_reviews():
    review_files = glob.glob('data_generation/raw_files/reviews/*.txt')

    rows = []
    for fpath in review_files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.txt', '').split('_')

        cid = parts[0]
        pid = parts[1]
        date = parts[2]

        with open(fpath, encoding='utf-8') as f:
            content = f.read()

        rows.append({
            'customer_id': cid,
            'product_id': pid,
            'review_date': date,
            'review_text': content,
            'file_path': fpath
        })

    df = pd.DataFrame(rows)
    df = add_bronze_meta(df, 'file_reviews_txt')
    append_bronze(df, 'reviews')


# ── 9. Images metadata ───────────────────────────────────────
def ingest_product_images():
    from PIL import Image as PILImage

    image_files = glob.glob('data_generation/raw_files/product_images/*.jpg')
    rows = []

    for fpath in image_files:
        pid = os.path.basename(fpath).replace('.jpg', '')
        size_bytes = os.path.getsize(fpath)

        is_corrupt = False
        width = height = None

        try:
            with PILImage.open(fpath) as img:
                width, height = img.size
        except Exception:
            is_corrupt = True

        rows.append({
            'product_id': pid,
            'file_path': fpath,
            'size_bytes': size_bytes,
            'width': width,
            'height': height,
            'is_corrupt': is_corrupt
        })

    df = pd.DataFrame(rows)
    df = add_bronze_meta(df, 'file_product_images')
    append_bronze(df, 'product_images_meta')


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Phase 2: Bronze Ingestion ===")
    print(f"Batch ID: {BATCH_ID}")

    ingest_mysql()
    ingest_postgres()
    ingest_product_catalogue()
    ingest_promotions()
    ingest_clickstream()
    ingest_returns()
    ingest_weekly_sales()
    ingest_reviews()
    ingest_product_images()

    print("\n✅ Bronze ingestion complete!")