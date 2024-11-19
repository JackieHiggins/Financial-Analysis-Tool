import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import datetime
from prediction_model import predict_future_transactions
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Financial Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['Transaction Amount'] = df['Transaction Amount'].astype(float)
    return df

df = load_data()

st.sidebar.header("Filters") # Sidebar Header
categories = ['All'] + sorted(df['Category'].unique().tolist()) # Categories
selected_category = st.sidebar.selectbox("Select Category", categories) # Select Category

months = pd.to_datetime(df['date']).dt.to_period('M').unique() # Months
month_options = ['All'] + [month.strftime('%B %Y') for month in sorted(months)] # Month Options
selected_month = st.sidebar.selectbox("Select Month", month_options) # Select Month

# Apply filters
filtered_df = df
if selected_category != 'All':  # Filter by Category
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]
if selected_month != 'All':  # Filter by Month
    selected_period = pd.Period(selected_month)
    filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.to_period('M') == selected_period]

col1, col2, col3, col4 = st.columns(4) # Columns for metrics
total_transactions = filtered_df['Transaction Amount'].sum()
average_transaction = filtered_df['Transaction Amount'].mean()
total_income = filtered_df[filtered_df['Transaction Amount'] > 0]['Transaction Amount'].sum()
total_expenses = abs(filtered_df[filtered_df['Transaction Amount'] < 0]['Transaction Amount'].sum())

col1.metric("Net Transaction Amount", f"${total_transactions:,.2f}")
col2.metric("Average Transaction Amount", f"${average_transaction:,.2f}")
col3.metric("Total Income", f"${total_income:,.2f}")
col4.metric("Total Expenses", f"${total_expenses:,.2f}")

# Daily transactions chart
daily_data = filtered_df.groupby('date')['Transaction Amount'].sum().reset_index()
chart_transactions = alt.Chart(daily_data).mark_line().encode(
    x='date:T',
    y='Transaction Amount:Q',
    tooltip=['date:T', 'Transaction Amount:Q']
).properties(
    title='Daily Transaction Amount Over Time'
)
st.altair_chart(chart_transactions, use_container_width=True)

# Load trusted anomalies
def load_trusted_anomalies():
    if os.path.exists('trusted_anomalies.csv') and os.path.getsize('trusted_anomalies.csv') > 0:
        return pd.read_csv('trusted_anomalies.csv')
    else:
        return pd.DataFrame(columns=['date', 'Category', 'Transaction Amount'])

# Save trusted anomaly
def save_trusted_anomaly(row):
    trusted_anomalies = load_trusted_anomalies()
    trusted_anomalies = pd.concat([trusted_anomalies, pd.DataFrame([row])], ignore_index=True)
    trusted_anomalies.to_csv('trusted_anomalies.csv', index=False)

# Detect anomalies, excluding trusted ones
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['Transaction Amount']])

    anomalies = df[df['anomaly'] == -1]
    trusted_anomalies = load_trusted_anomalies()

    # Ensure 'date' column is in datetime format for both DataFrames, inferring the format
    anomalies['date'] = pd.to_datetime(anomalies['date'], infer_datetime_format=True, errors='coerce')
    trusted_anomalies['date'] = pd.to_datetime(trusted_anomalies['date'], infer_datetime_format=True, errors='coerce')

    # Filter out trusted anomalies
    anomalies = anomalies.merge(
        trusted_anomalies,
        on=['date', 'Category', 'Transaction Amount'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop('_merge', axis=1)

    return anomalies



# Handle button clicks for trusting anomalies
def trust_transaction(transaction):
    save_trusted_anomaly(transaction)
    st.session_state["trusted_transactions"].append(transaction['date'])

# Initialize session state to track trusted transactions within the current session
if "trusted_transactions" not in st.session_state:
    st.session_state["trusted_transactions"] = []

st.header("Anomaly Detection")
anomalies_df = detect_anomalies(df)

if anomalies_df.empty:
    st.write("No anomalies detected in the current data.")
else:
    st.write("Anomalies detected in transactions:")

    for idx, row in anomalies_df.iterrows():
        if row['date'] not in st.session_state["trusted_transactions"]:
            st.write(f"Date: {row['date']}, Category: {row['Category']}, Amount: ${row['Transaction Amount']:.2f}")
            if st.button(f"Trust this transaction", key=f"trust_{idx}"):
                # Save trusted anomaly and update session state
                trust_transaction(row[['date', 'Category', 'Transaction Amount']])
                st.success("Transaction marked as trusted. It won't appear in future anomaly detections.")

# Financial Health Score
def calculate_financial_health(total_income, total_expenses):
    if total_income == 0:
        return 0
    savings_rate = (total_income - total_expenses) / total_income
    score = savings_rate * 100
    score = max(0, min(100, score))
    return score

financial_health_score = calculate_financial_health(total_income, total_expenses)

st.header("Financial Health Score")
st.metric("Your Financial Health Score", f"{financial_health_score:.2f}/100")

if financial_health_score > 70:
    st.success("You're in good financial health!")
elif financial_health_score > 40:
    st.warning("Your financial health is moderate. Consider reviewing your expenses.")
else:
    st.error("Your financial health is poor. Immediate attention needed!")

# Predictions with Prophet
future_income, future_expenditure = predict_future_transactions(df)

st.header("Predictions with Prophet")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Income (Prophet)")
    chart_predicted_income = alt.Chart(future_income).mark_line().encode(
        x='ds:T',
        y='yhat:Q',
        tooltip=['ds:T', 'yhat:Q']
    ).properties(
        title='Predicted Daily Income for Next 30 Days (Prophet)'
    )
    st.altair_chart(chart_predicted_income, use_container_width=True)

with col2:
    st.subheader("Predicted Expenditure (Prophet)")
    chart_predicted_expenditure = alt.Chart(future_expenditure).mark_line().encode(
        x='ds:T',
        y='yhat:Q',
        tooltip=['ds:T', 'yhat:Q']
    ).properties(
        title='Predicted Daily Expenditure for Next 30 Days (Prophet)'
    )
    st.altair_chart(chart_predicted_expenditure, use_container_width=True)
