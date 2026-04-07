import glob
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "dashboards" / "ecommerce_dashboard.duckdb"
GOLD_DIR = ROOT_DIR / "pipelines" / "gold" / "data"
SILVER_DIR = ROOT_DIR / "pipelines" / "silver" / "data"


def build_duckdb(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    try:
        kpi_tables = {
            "kpi_gmv": str(GOLD_DIR / "kpi_01_gmv" / "*.parquet"),
            "kpi_aov": str(GOLD_DIR / "kpi_08_aov" / "*.parquet"),
            "kpi_repeat_purchase_rate": str(GOLD_DIR / "kpi_09_repeat_purchase_rate" / "*.parquet"),
            "kpi_return_rate_by_cat": str(GOLD_DIR / "kpi_06_return_rate_by_category" / "*.parquet"),
            "kpi_inventory_turnover": str(GOLD_DIR / "kpi_07_inventory_turnover" / "*.parquet"),
        }

        silver_tables = {
            "silver_customers": str(SILVER_DIR / "customers" / "*.parquet"),
            "silver_orders": str(SILVER_DIR / "orders" / "*.parquet"),
            "silver_products": str(SILVER_DIR / "products" / "*.parquet"),
            "silver_inventory": str(SILVER_DIR / "inventory" / "*.parquet"),
            "silver_returns": str(SILVER_DIR / "returns" / "*.parquet"),
            "silver_clickstream": str(SILVER_DIR / "clickstream_events" / "*.parquet"),
            "silver_order_items": str(SILVER_DIR / "order_items" / "*.parquet"),
        }

        all_tables = {**kpi_tables, **silver_tables}

        for table_name, pattern in all_tables.items():
            files = glob.glob(pattern)
            if not files:
                print(f"  WARN {table_name}: no parquet files found")
                continue

            files_str = ", ".join(f"'{Path(file).as_posix()}'" for file in files)
            con.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS "
                f"SELECT * FROM read_parquet([{files_str}])"
            )
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  OK {table_name}: {count:,} rows")

        con.execute(
            """
            CREATE OR REPLACE VIEW dashboard_executive AS
            SELECT
                strftime(TRY_CAST(order_date AS DATE), '%Y-%m') AS month,
                SUM(total_amount) AS gmv,
                COUNT(DISTINCT order_id) AS order_count,
                AVG(total_amount) AS aov,
                COUNT(DISTINCT customer_id) AS active_customers
            FROM silver_orders
            WHERE status != 'CANCELLED'
            GROUP BY 1
            ORDER BY 1
            """
        )

        con.execute(
            """
            CREATE OR REPLACE VIEW dashboard_customer_ltv AS
            SELECT
                c.customer_id,
                c.segment,
                c.city,
                COUNT(DISTINCT o.order_id) AS order_count,
                SUM(o.total_amount) AS lifetime_value,
                AVG(o.total_amount) AS avg_order_value,
                CASE
                    WHEN SUM(o.total_amount) > 5000 THEN 'High'
                    WHEN SUM(o.total_amount) > 1000 THEN 'Mid'
                    ELSE 'Low'
                END AS ltv_tier
            FROM silver_customers c
            LEFT JOIN silver_orders o ON c.customer_id = o.customer_id
            GROUP BY 1, 2, 3
            """
        )

        con.execute(
            """
            CREATE OR REPLACE VIEW dashboard_inventory_health AS
            SELECT
                p.category_name,
                AVG(i.stock_qty) AS avg_stock,
                SUM(CASE WHEN i.is_writeoff THEN 1 ELSE 0 END) AS writeoff_count,
                COUNT(*) AS total_skus
            FROM silver_inventory i
            JOIN silver_products p ON i.product_id = p.product_id
            GROUP BY 1
            ORDER BY avg_stock DESC
            """
        )
    finally:
        con.close()

    print(f"\nDuckDB created at {db_path}")
    print("Now connect Superset to this file using the DuckDB SQLAlchemy driver.")
    print(f"Connection string: duckdb:///{db_path.as_posix()}")
    return db_path


if __name__ == "__main__":
    build_duckdb()
