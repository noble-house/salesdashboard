# sales_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
import openai
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import io
import base64
from xhtml2pdf import pisa
from dotenv import load_dotenv
import os

# Load environment variables for OpenAI Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Excel data
@st.cache_data
def load_data():
    xls = pd.ExcelFile("Sales_AI_Analysis_Report.xlsx")
    customer_product = pd.read_excel(xls, 'Customer_Product_Yearly')
    monthly = pd.read_excel(xls, 'Monthly_Trends')
    total_rev = pd.read_excel(xls, 'Total_Revenue_Share')
    invoices = pd.read_excel(xls, 'Invoices_Per_Customer')
    return customer_product, monthly, total_rev, invoices

customer_product, monthly_raw, total_rev_raw, invoices = load_data()

# Create main filters
all_years = sorted(customer_product['Year'].dropna().unique())
all_products = sorted(customer_product['Product'].dropna().unique())
all_customers = sorted(customer_product['Particulars'].dropna().unique())

for key, default in {"years": all_years, "products": all_products, "customers": all_customers}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Create 4 tabs as per new structure
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Management Insights", "📈 Forecasting & Segmentation", "🤖 AI Summary"])

# =============== TAB 1 — Dashboard (Existing) ===============
with tab1:
    st.title("📊 AI-Powered Sales Dashboard")

    with st.expander("🔍 Filter Data"):
        col1, col2, col3 = st.columns([1, 2, 2])
        for key, label, values in zip(["years", "products", "customers"],
                                       ["📅 Year", "📦 Product", "👥 Customer"],
                                       [all_years, all_products, all_customers]):
            with col1 if key == "years" else col2 if key == "products" else col3:
                st.markdown(f"**{label} Filter**")
                toggle = st.checkbox(f"Select/Deselect All {label}s", key=f"toggle_{key}")
                if toggle:
                    st.session_state[key] = [] if len(st.session_state[key]) == len(values) else values
                for val in values:
                    if st.checkbox(val, value=val in st.session_state[key], key=f"{key}_{val}"):
                        if val not in st.session_state[key]:
                            st.session_state[key].append(val)
                    else:
                        if val in st.session_state[key]:
                            st.session_state[key].remove(val)

    filtered = customer_product[
        customer_product['Year'].isin(st.session_state["years"]) &
        customer_product['Product'].isin(st.session_state["products"]) &
        customer_product['Particulars'].isin(st.session_state["customers"])
    ]

    monthly_filtered = monthly_raw.copy()
    monthly_filtered['Month'] = pd.to_datetime(monthly_filtered['Month'], errors='coerce')
    selected_year_nums = [int(yr.split("-")[0]) for yr in st.session_state["years"]]
    monthly_filtered = monthly_filtered[monthly_filtered['Month'].dt.year.isin(selected_year_nums)]

    total_sales = filtered['Gross Total'].sum()
    total_customers = filtered['Particulars'].nunique()
    total_invoices = invoices[
        invoices['Year'].isin(st.session_state["years"]) &
        invoices['Particulars'].isin(st.session_state["customers"])
    ]['Invoice Count'].sum()

    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<h5>💰 Total Revenue<br/><span style='font-size:20px;'>₹{total_sales:,.0f}</span></h5>", unsafe_allow_html=True)
    k2.markdown(f"<h5>👥 Unique Customers<br/><span style='font-size:20px;'>{total_customers}</span></h5>", unsafe_allow_html=True)
    k3.markdown(f"<h5>🧾 Invoice Count<br/><span style='font-size:20px;'>{int(total_invoices)}</span></h5>", unsafe_allow_html=True)

    st.subheader("📦 Revenue by Product")
    st.caption("This chart shows the total revenue contribution by each product.")
    if not filtered.empty:
        fig1 = px.bar(filtered.groupby("Product")['Gross Total'].sum().reset_index(), 
                      x="Product", y="Gross Total", text_auto=True)
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📅 Monthly Revenue Trend")
    st.caption("This chart shows how revenue has changed month-over-month for selected years.")
    if not monthly_filtered.empty:
        fig2 = px.line(monthly_filtered, x="Month", y="Gross Total", title="Monthly Revenue Trend")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("👥 Top 10 Customers")
    st.caption("These are the top customers ranked by total revenue generated.")
    top_customers = filtered.groupby("Particulars")['Gross Total'].sum().reset_index().sort_values(by='Gross Total', ascending=False).head(10)
    if not top_customers.empty:
        fig3 = px.pie(top_customers, names="Particulars", values="Gross Total", title="Top 10 Customers")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📦 Top 10 Products")
    st.caption("These are the products that generated the highest revenue.")
    top_products = filtered.groupby("Product")['Gross Total'].sum().reset_index().sort_values(by='Gross Total', ascending=False).head(10)
    if not top_products.empty:
        fig4 = px.bar(top_products, x="Product", y="Gross Total", text_auto=True)
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📄 Filtered Sales Table")
    st.dataframe(filtered, use_container_width=True)
# ==== NEW Tab 2: Sales Deep Dive Analysis with Visuals ====
with tab2:
    st.title("📊 Sales Deep Dive Analysis")

    # 1️⃣ Customer Ranking across 3 years
    st.header("📈 Customer Ranking Year-wise")
    customer_rank = customer_product.groupby(['Year', 'Particulars'])['Gross Total'].sum().reset_index()
    customer_rank['Rank'] = customer_rank.groupby('Year')['Gross Total'].rank(ascending=False, method='first')
    customer_rank_pivot = customer_rank.pivot(index='Particulars', columns='Year', values='Rank').fillna('-').astype(str)
    st.dataframe(customer_rank_pivot.reset_index(), use_container_width=True)

    # 2️⃣ Pareto Analysis - Table
    st.header("📊 Pareto Customer Contribution")
    pareto = customer_product.groupby('Particulars')['Gross Total'].sum().sort_values(ascending=False)
    pareto_cumsum = pareto.cumsum() / pareto.sum()
    thresholds = [0.5, 0.75, 0.90]
    threshold_counts = {f"{int(t*100)}% Sales": (pareto_cumsum <= t).sum() for t in thresholds}
    st.write(pd.DataFrame.from_dict(threshold_counts, orient='index', columns=["# Companies"]))

    # 2️⃣ Pareto Chart
    st.subheader("Pareto Distribution Chart")
    pareto_df = pd.DataFrame({'Customer': pareto.index, 'Revenue': pareto.values, 'Cumulative %': pareto_cumsum.values * 100}).reset_index(drop=True)
    fig_pareto = px.bar(pareto_df, x='Customer', y='Revenue')
    fig_pareto.add_scatter(x=pareto_df['Customer'], y=pareto_df['Cumulative %'], mode='lines+markers', name='Cumulative %', yaxis='y2')
    fig_pareto.update_layout(
        yaxis=dict(title='Revenue'),
        yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
        xaxis_title='Customer', title="Pareto Distribution"
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    # 3️⃣ Product Contribution by Year Pivot
    st.header("📊 Product Contribution by Year and Overall")
    product_year = customer_product.groupby(['Year', 'Product'])['Gross Total'].sum().reset_index()
    product_pivot = product_year.pivot_table(index='Product', columns='Year', values='Gross Total', fill_value=0)
    product_pivot['Total'] = product_pivot.sum(axis=1)
    product_pivot = product_pivot.sort_values('Total', ascending=False)
    st.dataframe(product_pivot.reset_index(), use_container_width=True)

    st.subheader("📊 Product Sales Heatmap")
    fig_heatmap = px.imshow(product_pivot.drop(columns='Total'), text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # 4️⃣ Product Share within each year
    st.subheader("Product Sales Share % (Year-wise)")
    product_year['Yearly Share %'] = product_year.groupby('Year')['Gross Total'].transform(lambda x: 100 * x / x.sum())
    st.dataframe(product_year, use_container_width=True)

    # 5️⃣ New Customers in 2023-24
    st.header("🆕 New Customers in 2023-24")
    df_new = customer_product.pivot_table(index='Particulars', columns='Year', values='Gross Total', fill_value=0).reset_index()
    year_cols = df_new.columns.tolist()
    has_all_years = all(y in year_cols for y in ['2021-22', '2022-23', '2023-24'])
    if has_all_years:
        df_new_2023 = df_new[(df_new['2021-22'] == 0) & (df_new['2022-23'] == 0) & (df_new['2023-24'] > 0)]
        st.dataframe(df_new_2023, use_container_width=True)
    else:
        st.warning("Not all years present for this analysis.")

    # 6️⃣ Dormant Customers - No business in latest year
    st.header("🛑 Dormant Customers (No Sales in 2023-24)")
    if '2023-24' in df_new.columns:
        dormant = df_new[df_new['2023-24'] == 0]
        st.dataframe(dormant, use_container_width=True)
    else:
        st.warning("2023-24 data not found")

    # 7️⃣ Customers with >20% Growth or Drop YoY
    st.header("📊 YoY Sales Change >20%")
    for year in ['2022-23', '2023-24']:
        start_year = int(year.split('-')[0])
        prev_start_year = start_year - 1
        prev_year = f"{prev_start_year}-{str(prev_start_year + 1)[-2:]}"
        if prev_year in df_new.columns and year in df_new.columns:
            change_col = f"Change_{year}"
            df_new[change_col] = df_new.apply(
                lambda row: ((row[year] - row[prev_year]) / row[prev_year] * 100) if row[prev_year] > 0 else np.nan, axis=1
            )

    if 'Change_2023-24' in df_new.columns:
        df_growth = df_new[df_new['Change_2023-24'] > 20].copy()
        df_growth['Change_2023-24'] = df_growth['Change_2023-24'].apply(lambda x: f"+{x:.1f}%" if pd.notnull(x) else "-")
        st.subheader("Companies with >20% Sales Growth YoY")
        st.dataframe(df_growth[['Particulars', '2021-22', '2022-23', '2023-24', 'Change_2023-24']], use_container_width=True)

        df_drop = df_new[df_new['Change_2023-24'] < -20].copy()
        df_drop['Change_2023-24'] = df_drop['Change_2023-24'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
        st.subheader("Companies with >20% Sales Drop YoY")
        st.dataframe(df_drop[['Particulars', '2021-22', '2022-23', '2023-24', 'Change_2023-24']], use_container_width=True)

    # 8️⃣ Product-wise Customer Count
    st.header("📊 Product Adoption Spread")
    product_spread = customer_product.groupby(['Year', 'Product'])['Particulars'].nunique().reset_index()
    product_spread = product_spread.rename(columns={'Particulars': 'Customer Count'})
    st.dataframe(product_spread, use_container_width=True)

    # 9️⃣ Product Share of Revenue per Year
    st.header("📊 Product Share of Revenue")
    yearly_sales = customer_product.groupby(['Year'])['Gross Total'].sum().reset_index()
    product_year_total = product_year.merge(yearly_sales, on='Year', suffixes=('', '_Total'))
    product_year_total['Product % Share'] = product_year_total['Gross Total'] / product_year_total['Gross Total_Total'] * 100
    st.dataframe(product_year_total[['Year', 'Product', 'Gross Total', 'Product % Share']], use_container_width=True)

    # 🔟 Customer-wise Product Portfolio
    st.header("📊 Customer-wise Product Portfolio (%)")
    customer_product_pivot = customer_product.pivot_table(index="Particulars", columns="Product", values="Gross Total", aggfunc="sum", fill_value=0)
    customer_product_pivot_pct = customer_product_pivot.div(customer_product_pivot.sum(axis=1), axis=0) * 100
    st.dataframe(customer_product_pivot_pct.round(1), use_container_width=True)

    # 🔄 Shifting Product Trends
    st.header("📈 Shifting Product Trends (YoY %)")
    product_trend = customer_product.pivot_table(index="Product", columns="Year", values="Gross Total", aggfunc="sum", fill_value=0).reset_index()
    for year in ['2022-23', '2023-24']:
        start_year = int(year.split('-')[0])
        prev_year = f"{start_year-1}-{str(start_year)[-2:]}"
        if prev_year in product_trend.columns and year in product_trend.columns:
            product_trend[f"Change_{year}"] = ((product_trend[year] - product_trend[prev_year]) / product_trend[prev_year].replace(0, np.nan)) * 100
    st.dataframe(product_trend.fillna("-"), use_container_width=True)

   ### 🔄 Improved Product Trend Heatmap with Better Coloring

st.header("📈 Product Trend Heatmap (Improved Growth %)")

# Calculate product-level YoY % change like before
product_trend = customer_product.pivot_table(index="Product", columns="Year", values="Gross Total", aggfunc="sum", fill_value=0).reset_index()

for year in ['2022-23', '2023-24']:
    start_year = int(year.split('-')[0])
    prev_year = f"{start_year-1}-{str(start_year)[-2:]}"
    if prev_year in product_trend.columns and year in product_trend.columns:
        product_trend[f"Change_{year}"] = ((product_trend[year] - product_trend[prev_year]) / product_trend[prev_year].replace(0, np.nan)) * 100

# Subset only the % growth columns
growth_cols = [col for col in product_trend.columns if 'Change_' in col]
growth_data = product_trend[growth_cols].fillna(0).clip(-100, 100)  # cap to [-100%, 100%]

# Create heatmap with better scaling
fig_heatmap = px.imshow(
    growth_data,
    text_auto=".1f",  # rounded text values
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-100, zmax=100,  # fix color scale
    labels=dict(color="Growth %")
)

fig_heatmap.update_layout(
    title="Product Trend Heatmap (Capped at ±100%)",
    xaxis_title="Year",
    yaxis_title="Product Index",
    coloraxis_colorbar=dict(
        title="% Change",
        ticksuffix="%"
    )
)

st.plotly_chart(fig_heatmap, use_container_width=True)



# ================== TAB 3 — Forecasting & Segmentation ==================

with tab3:
    st.title("📈 Forecasting & Segmentation Insights")

    st.header("📊 12-Month Revenue Forecast")
    st.markdown("Forecasts your expected revenue for the next 12 months based on past trends. "
                "Helps in setting sales targets and financial planning.")

    monthly_forecast = monthly_raw.copy()
    monthly_forecast['Month'] = pd.to_datetime(monthly_forecast['Month'], errors='coerce')
    monthly_forecast = monthly_forecast.groupby('Month')['Gross Total'].sum().reset_index()
    monthly_forecast = monthly_forecast.rename(columns={'Month': 'ds', 'Gross Total': 'y'})

    try:
        model = Prophet()
        model.fit(monthly_forecast)
        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)

        fig = px.line()
        fig.add_scatter(x=monthly_forecast['ds'], y=monthly_forecast['y'], mode='lines', name='Actual')
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
        fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
        fig.update_layout(title="12-Month Revenue Forecast with Confidence Intervals")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")

    st.header("👥 Customer Segmentation")
    st.markdown("Groups your customers into clusters like 'High Value', 'Occasional', 'Dormant', or 'At Risk' "
                "based on revenue and invoice volume. Useful for targeted engagement strategies.")
    try:
        revenue_df = customer_product.groupby("Particulars")["Gross Total"].sum().reset_index()
        invoice_df = invoices.groupby("Particulars")["Invoice Count"].sum().reset_index()
        cluster_data = pd.merge(revenue_df, invoice_df, on="Particulars", how="inner")

        km = KMeans(n_clusters=4, random_state=42)
        cluster_data["Cluster"] = km.fit_predict(cluster_data[["Gross Total", "Invoice Count"]])
        cluster_map = {0: "High Value", 1: "Occasional", 2: "Dormant", 3: "At Risk"}
        cluster_data["Segment"] = cluster_data["Cluster"].map(cluster_map)

        fig_cluster = px.scatter(cluster_data, x="Gross Total", y="Invoice Count", color="Segment", hover_data=["Particulars"],
                                 title="Customer Segments (Revenue vs Invoices)")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("📆 Segment-wise Revenue Trend")
        st.markdown("Shows how each customer segment contributed to total revenue annually — useful for prioritizing focus areas.")
        merged = pd.merge(customer_product, cluster_data[["Particulars", "Segment"]], on="Particulars", how="inner")
        trend = merged.groupby(['Year', 'Segment'])['Gross Total'].sum().reset_index()
        fig_trend = px.line(trend, x="Year", y="Gross Total", color="Segment", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ================== TAB 4 — AI Summary with GPT and Export ==================

from xhtml2pdf import pisa

with tab4:
    st.title("🤖 AI Summary")
    st.caption("This section uses GPT-4 to provide rich insights based on real revenue and product performance data.")

    # Refresh Button
    refresh = st.button("🔁 Refresh Insights")

    @st.cache_data(show_spinner=False)
    def generate_ai_summary():
        monthly_trend_df = monthly_raw.copy()
        monthly_trend_df['Month'] = pd.to_datetime(monthly_trend_df['Month'], errors='coerce')
        monthly_trend_df = monthly_trend_df.groupby('Month')['Gross Total'].sum().reset_index()
        recent_months = monthly_trend_df.sort_values(by='Month').tail(12)

        month_revenue_text = "\n".join([
            f"{row['Month'].strftime('%b %Y')}: ₹{row['Gross Total']:,.0f}" for _, row in recent_months.iterrows()
        ])

        product_revenue = customer_product.groupby("Product")["Gross Total"].sum().reset_index()
        product_revenue = product_revenue.sort_values(by="Gross Total", ascending=False)

        top_3 = product_revenue.head(3)
        bottom_3 = product_revenue.tail(3)

        top_products_text = "\n".join([
            f"{row['Product']}: ₹{row['Gross Total']:,.0f}" for _, row in top_3.iterrows()
        ])
        bottom_products_text = "\n".join([
            f"{row['Product']}: ₹{row['Gross Total']:,.0f}" for _, row in bottom_3.iterrows()
        ])

        prompt = f"""
You are a senior sales analyst AI. Analyze the following data and generate a smart, executive-level summary in markdown:

📊 Monthly Revenue Trend (Last 12 Months):
{month_revenue_text}

🏆 Top 3 Performing Products:
{top_products_text}

⚠️ Bottom 3 Performing Products:
{bottom_products_text}

Your analysis must include:
- **Key Revenue Trends** (based on month-over-month patterns)
- **Insights on Best/Worst Products**
- **Potential Customer Behavior Patterns**
- **Strategic Recommendations for Business Growth**

Be concise yet insightful. Avoid vague or hypothetical statements.
"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful sales analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    if "ai_summary_output" not in st.session_state or refresh:
        try:
            st.session_state.ai_summary_output = generate_ai_summary()
        except Exception as e:
            st.error(f"AI Summary failed: {e}")

    # 📈 Monthly Revenue Chart
    monthly_trend_df = monthly_raw.copy()
    monthly_trend_df['Month'] = pd.to_datetime(monthly_trend_df['Month'], errors='coerce')
    monthly_trend_df = monthly_trend_df.groupby('Month')['Gross Total'].sum().reset_index()
    fig = px.line(monthly_trend_df, x='Month', y='Gross Total', title="📈 Monthly Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

    # 🏆 Top Products Table
    st.subheader("🏆 Top 10 Products by Revenue")
    top_products_df = customer_product.groupby("Product")["Gross Total"].sum().reset_index()
    top_products_df = top_products_df.sort_values(by="Gross Total", ascending=False).head(10)
    st.dataframe(top_products_df, use_container_width=True)

    # 🧠 AI Summary Markdown
    if "ai_summary_output" in st.session_state:
        st.subheader("🧠 GPT-4 Executive Summary")
        st.markdown(st.session_state.ai_summary_output)

        # Chart image conversion for PDF
        def get_chart_base64():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(monthly_trend_df['Month'], monthly_trend_df['Gross Total'], marker='o')
            ax.set_title("Monthly Revenue Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Revenue (INR)")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            return f'<img src="data:image/png;base64,{img_base64}" width="600"/>'

        # Full HTML to PDF generator
        def convert_full_html_to_pdf(markdown_text, chart_html, top_table_df, file_path):
            table_html = top_table_df.to_html(index=False, border=1)
            full_html = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 30px; font-size: 14px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    table, th, td {{ border: 1px solid #999; padding: 8px; }}
                </style>
            </head>
            <body>
                <h2>Executive AI Summary</h2>
                <p>{markdown_text.replace('\n', '<br>')}</p>
                <hr>
                <h2>📈 Monthly Revenue Chart</h2>
                {chart_html}
                <hr>
                <h2>🏆 Top 10 Products</h2>
                {table_html}
            </body>
            </html>
            """
            with open(file_path, "w+b") as f:
                pisa.CreatePDF(full_html, dest=f)

        # Export as PDF button
        st.subheader("📤 Export Full AI Report as PDF")
        if st.button("📄 Export AI Summary with Chart + Table"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                chart_html = get_chart_base64()
                convert_full_html_to_pdf(
                    markdown_text=st.session_state.ai_summary_output,
                    chart_html=chart_html,
                    top_table_df=top_products_df,
                    file_path=tmpfile.name
                )
                with open(tmpfile.name, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Full AI Report PDF",
                        data=f,
                        file_name="AI_Sales_Summary.pdf",
                        mime="application/pdf"
                    )
