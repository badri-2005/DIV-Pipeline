# data_generation/generate_data.py
import os, json, random, shutil, hashlib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from PIL import Image
import openpyxl

load_dotenv()
fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Master ID pools ──────────────────────────────────────────────
customer_ids = [f'CUST-{str(i).zfill(6)}' for i in range(1, 5001)]
product_ids  = [f'PROD-{str(i).zfill(5)}' for i in range(1, 1001)]
order_ids    = [f'ORD-{str(i).zfill(8)}' for i in range(20240001, 20250001)]
promo_ids    = [f'PROMO-{str(i).zfill(4)}' for i in range(1, 101)]
category_ids = list(range(1, 21))
warehouse_ids = [f'WH-{str(i).zfill(3)}' for i in range(1, 11)]

def get_mysql_engine():
    url = (f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
           f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}")
    return create_engine(url)

def generate_customers(n=5000):
    rows = []
    emails_used = set()
    for i, cid in enumerate(customer_ids):
        name = fake.name()
        # Noise 1: Mixed case names (10% all-upper, 5% all-lower)
        if i % 20 == 0:   name = name.upper()
        elif i % 20 == 1: name = name.lower()

        email = fake.email()
        # Noise 2: 5% duplicate emails
        if i % 20 < 1 and emails_used:
            email = random.choice(list(emails_used))
        emails_used.add(email)

        city = fake.city()
        # Noise 3: 8% null city
        if random.random() < 0.08: city = None

        signup = fake.date_between(start_date='-5y', end_date='today')
        # Noise 4: 10% dates as DD-MM-YYYY string instead of proper date
        if random.random() < 0.10:
            signup = signup.strftime('%d-%m-%Y')  # string, not date
        else:
            signup = signup.isoformat()

        rows.append({
            'customer_id': cid,
            'full_name': name,
            'email': email,
            'city': city,
            'signup_date': signup,
            'segment': random.choice(['premium', 'regular', 'occasional'])
        })
    return pd.DataFrame(rows)

def generate_orders(n=10000):
    STATUS_VARIANTS = {
        'PLACED':    ['placed', 'Placed', 'order_placed', 'new', 'PLACED', 'pending'],
        'SHIPPED':   ['shipped', 'Shipped', 'dispatched', 'in_transit', 'SHIPPED', 'out_for_delivery'],
        'CANCELLED': ['cancelled', 'Canceled', 'CANCELLED', 'cancel', 'void', 'VOID'],
    }
    rows = []
    for oid in order_ids[:n]:
        cid = random.choice(customer_ids)
        status_group = random.choice(list(STATUS_VARIANTS.keys()))
        status = random.choice(STATUS_VARIANTS[status_group])  # Noise: 6 variants per state

        order_date = fake.date_between(start_date='-1y', end_date='today')
        ship_date  = order_date + timedelta(days=random.randint(1, 7))

        # Noise: 12% null ship_date
        if random.random() < 0.12: ship_date = None

        amount = round(random.uniform(100, 15000), 2)
        # Noise: 8% amount as string with commas
        if random.random() < 0.08:
            amount = f"{amount:,.2f}"  # "1,234.56"

        promo = random.choice(promo_ids + [None, None, None])  # 75% no promo

        rows.append({
            'order_id': oid,
            'customer_id': cid,
            'order_date': order_date.isoformat(),
            'status': status,
            'total_amount': amount,
            'ship_date': ship_date.isoformat() if ship_date else None,
            'promo_id': promo
        })
    return pd.DataFrame(rows)

def generate_order_items():
    rows = []
    for oid in order_ids:
        for _ in range(random.randint(1, 5)):
            rows.append({
                'order_item_id': fake.uuid4(),
                'order_id': oid,
                'product_id': random.choice(product_ids),
                'quantity': random.randint(1, 10),
                'unit_price': round(random.uniform(50, 5000), 2)
            })
    return pd.DataFrame(rows)

def seed_mysql():
    engine = get_mysql_engine()
    print("Seeding MySQL...")
    generate_customers().to_sql('customers',   engine, if_exists='replace', index=False)
    generate_orders().to_sql('orders',         engine, if_exists='replace', index=False)
    generate_order_items().to_sql('order_items', engine, if_exists='replace', index=False)
    print("  ✓ customers, orders, order_items written")
    
def get_pg_engine():
    url = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
           f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}")
    return create_engine(url)

def generate_products():
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports',
                  'Toys', 'Food', 'Beauty', 'Furniture', 'Accessories']
    rows = []
    for pid in product_ids:
        desc = fake.sentence(nb_words=12)
        # Noise: HTML tags in description
        if random.random() < 0.15:
            desc = f"<b>{desc}</b> <br/> {fake.sentence()}"

        name = fake.catch_phrase()
        # Noise: 15% trailing whitespace
        if random.random() < 0.15: name = name + "   "

        cat_id = random.choice(category_ids)
        # Noise: 8% null category_id
        if random.random() < 0.08: cat_id = None

        rows.append({
            'product_id': pid,
            'product_name': name,
            'description': desc,
            'category_id': cat_id,
            'category_name': random.choice(categories),
            'price': round(random.uniform(50, 10000), 2)
        })
    return pd.DataFrame(rows)

def generate_inventory():
    rows = []
    for pid in product_ids:
        wid = random.choice(warehouse_ids)
        # Noise: 20% warehouse_id with trailing space
        if random.random() < 0.20: wid = wid + " "

        qty = random.randint(0, 500)
        # Noise: 3% negative stock_qty
        if random.random() < 0.03: qty = -random.randint(1, 50)

        rows.append({
            'inventory_id': fake.uuid4(),
            'product_id': pid,
            'warehouse_id': wid,
            'stock_qty': qty,
            'last_updated': fake.date_between(start_date='-30d', end_date='today').isoformat()
        })
    return pd.DataFrame(rows)

def seed_postgres():
    engine = get_pg_engine()
    print("Seeding PostgreSQL...")
    generate_products().to_sql('products',   engine, if_exists='replace', index=False)
    generate_inventory().to_sql('inventory', engine, if_exists='replace', index=False)
    print("  ✓ products, inventory written")
    
def generate_product_catalogue_csv(path='data_generation/raw_files/product_catalogue.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    null_variants = ['', 'N/A', 'NULL']
    for pid in product_ids[:500]:
        price = round(random.uniform(50, 8000), 2)
        stock = random.randint(0, 200)
        # Noise: mixed null types
        if random.random() < 0.03: price = random.choice(null_variants)
        if random.random() < 0.03: stock = random.choice(null_variants)

        rows.append({
            'product_id': pid,
            'product_name': fake.catch_phrase(),
            'price': price,
            'stock': stock,
            'supplier': fake.company()
        })

    df = pd.DataFrame(rows)
    # Noise: mixed comma+semicolon delimiters (write some rows with ; separator)
    with open(path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig adds BOM
        for i, row in df.iterrows():
            delimiter = ';' if i % 10 == 0 else ','
            f.write(delimiter.join(str(v) for v in row.values) + '\n')
    print(f"  ✓ {path}")

def generate_clickstream_json(path='data_generation/raw_files/clickstream_events.json'):
    events = []
    event_types = ['view', 'click', 'add_to_cart', 'checkout', 'purchase']
    typo_map = {'view': 'veiw', 'click': 'clck'}  # Noise: typos
    for _ in range(50000):
        et = random.choice(event_types)
        # Noise: 20% typo in event_type
        if random.random() < 0.20 and et in typo_map:
            et = typo_map[et]

        # Noise: mixed Unix+ISO timestamps
        ts = fake.date_time_between(start_date='-6m', end_date='now')
        timestamp = int(ts.timestamp()) if random.random() < 0.5 else ts.isoformat()

        session_id = fake.uuid4()
        # Noise: 5% null session_id
        if random.random() < 0.05: session_id = None

        events.append({
            'event_id': fake.uuid4(),
            'customer_id': random.choice(customer_ids),
            'product_id': random.choice(product_ids),
            'event_type': et,
            'session_id': session_id,
            'timestamp': timestamp,
            'cart_value': round(random.uniform(100, 5000), 2) if et in ['add_to_cart', 'checkout'] else None
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"  ✓ {path}")

def generate_returns_json(path='data_generation/raw_files/returns.json'):
    returns = []
    for _ in range(2000):
        oid = random.choice(order_ids)
        # Noise: 15% missing order_id
        if random.random() < 0.15: oid = None

        amount = round(random.uniform(50, 3000), 2)
        # Noise: 10% amount as 'Rs.1200' string
        if random.random() < 0.10: amount = f"Rs.{amount}"

        returns.append({
            'return_id': fake.uuid4(),
            'order_id': oid,
            'customer_id': random.choice(customer_ids),
            'return_date': fake.date_between(start_date='-6m', end_date='today').isoformat(),
            'refund_amount': amount,
            'reason': random.choice(['Defective', 'Wrong size', 'Not as described', 'Changed mind'])
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(returns, f, indent=2)
    print(f"  ✓ {path}")

def generate_promotions_csv(path='data_generation/raw_files/promotions.csv'):
    rows = []
    base_date = datetime(2024, 1, 1)
    for pid in promo_ids:
        start = base_date + timedelta(days=random.randint(0, 300))
        end   = start + timedelta(days=random.randint(3, 30))
        discount = random.randint(5, 50)
        # Noise: 20% discount as string '15%' instead of int
        if random.random() < 0.20: discount = f"{discount}%"
        rows.append({
            'promo_id': pid,
            'promo_name': f"PROMO_{pid}",
            'start_date': start.date().isoformat(),
            'end_date': end.date().isoformat(),   # Overlapping is natural
            'discount_pct': discount,
            'promo_code': fake.bothify(text='??##??').upper()
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ {path}")

def generate_weekly_sales_excel(path='data_generation/raw_files/weekly_sales_summary.xlsx'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = openpyxl.Workbook()
    for week_num in range(1, 5):
        ws = wb.create_sheet(title=f"Week_{week_num}")
        # Noise: varying column headers per sheet
        if week_num % 2 == 0:
            headers = ['ProductID', 'ProductName', 'UnitsSold', 'Revenue', 'Returns']
        else:
            headers = ['product_id', 'product_name', 'units_sold', 'revenue', 'returns']
        ws.append(headers)
        for pid in random.sample(product_ids, 20):
            ws.append([pid, fake.catch_phrase(),
                       random.randint(1, 100),
                       round(random.uniform(1000, 50000), 2),
                       random.randint(0, 10)])
        # Noise: embed subtotal row as data
        ws.append([None, 'SUBTOTAL', '=SUM(C2:C21)', '=SUM(D2:D21)', '=SUM(E2:E21)'])
    del wb['Sheet']
    wb.save(path)
    print(f"  ✓ {path}")

def generate_review_txts(base_dir='data_generation/raw_files/reviews'):
    os.makedirs(base_dir, exist_ok=True)
    for _ in range(500):
        cid = random.choice(customer_ids)
        pid = random.choice(product_ids)
        date = fake.date_between(start_date='-1y', end_date='today').isoformat()
        fname = f"{base_dir}/{cid}_{pid}_{date}.txt"
        # Noise: HTML entities, emoji, multilingual words
        review = fake.paragraph(nb_sentences=4)
        review = review.replace("&", "&amp;").replace("<", "&lt;")
        if random.random() < 0.3:
            review += " 😊👍 Très bien! बहुत अच्छा"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(review)
    print(f"  ✓ {base_dir}/ (500 txt files)")

def generate_product_images(base_dir='data_generation/raw_files/product_images'):
    os.makedirs(base_dir, exist_ok=True)
    for pid in product_ids[:100]:  # 100 images to keep it manageable
        w = random.randint(100, 800)
        h = random.randint(100, 800)
        img = Image.new('RGB', (w, h), color=(
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        fpath = f"{base_dir}/{pid}.jpg"
        # Noise: 3% corrupted (truncate bytes)
        if random.random() < 0.03:
            import io
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            with open(fpath, 'wb') as f:
                f.write(buf.getvalue()[:len(buf.getvalue())//2])  # half the bytes
        else:
            img.save(fpath, 'JPEG')
    print(f"  ✓ {base_dir}/ (100 jpg files, 3% corrupted)")

# ── Main runner ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Phase 1: Data Generation ===")
    seed_mysql()
    seed_postgres()
    generate_product_catalogue_csv()
    generate_promotions_csv()
    generate_clickstream_json()
    generate_returns_json()
    generate_weekly_sales_excel()
    generate_review_txts()
    generate_product_images()
    print("\n✅ All sources generated successfully!")