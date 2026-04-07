from pathlib import Path

import duckdb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from create_duckdb import DB_PATH, build_duckdb

OUTPUT_PATH = Path(__file__).resolve().parent / "dashboard.html"

def dashboard_views_exist() -> bool:
    if not DB_PATH.exists():
        return False

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        count = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.views
            WHERE table_name IN (
                'dashboard_executive',
                'dashboard_customer_ltv',
                'dashboard_inventory_health'
            )
            """
        ).fetchone()[0]
        return count == 3
    finally:
        con.close()


if not dashboard_views_exist():
    build_duckdb(DB_PATH)

con = duckdb.connect(str(DB_PATH), read_only=True)

# Fetch dashboard data
exec_df = con.execute("SELECT * FROM dashboard_executive ORDER BY month").df()
ltv_df = con.execute(
    """
    SELECT ltv_tier, COUNT(*) AS cnt, AVG(lifetime_value) AS avg_ltv
    FROM dashboard_customer_ltv
    GROUP BY 1
    """
).df()
inv_df = con.execute(
    "SELECT * FROM dashboard_inventory_health ORDER BY avg_stock DESC LIMIT 10"
).df()
ret_df = con.execute(
    "SELECT * FROM kpi_return_rate_by_cat ORDER BY return_rate DESC"
).df()

fig = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "xy"}, {"type": "domain"}], [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(
        "GMV trend (monthly)",
        "Customer LTV tier distribution",
        "Average stock by category (top 10)",
        "Return rate by category",
    ),
)

fig.add_trace(
    go.Scatter(
        x=exec_df["month"],
        y=exec_df["gmv"],
        mode="lines+markers",
        fill="tozeroy",
        name="GMV",
        line=dict(color="#1D9E75", width=2),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Pie(
        labels=ltv_df["ltv_tier"],
        values=ltv_df["cnt"],
        marker_colors=["#5DCAA5", "#7F77DD", "#EF9F27"],
        name="LTV tier",
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Bar(
        x=inv_df["category_name"],
        y=inv_df["avg_stock"],
        marker_color="#185FA5",
        name="Avg stock",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=ret_df["category_name"],
        y=ret_df["return_rate"] * 100,
        marker_color="#D85A30",
        name="Return rate %",
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=ret_df["category_name"],
        y=[15] * len(ret_df),
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="15% threshold",
    ),
    row=2,
    col=2,
)

fig.update_layout(
    height=700,
    title_text="E-commerce Gold Dashboard",
    showlegend=False,
    template="plotly_white",
)
fig.write_html(str(OUTPUT_PATH))
fig.show()
con.close()
print(f"Dashboard saved to {OUTPUT_PATH}")
