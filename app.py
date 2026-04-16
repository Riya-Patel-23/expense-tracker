import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Expense Tracker", layout="wide")

st.title("Smart Expense Tracker Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("expenses.csv")

data['Date'] = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
data = data.dropna(subset=['Date'])

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("Add Expense")

date = st.sidebar.date_input("Date")
category = st.sidebar.selectbox(
    "Category",
    ["Food", "Travel", "Shopping", "Bills", "Entertainment"]
)
amount = st.sidebar.number_input("Amount", min_value=0)
payment = st.sidebar.selectbox(
    "Payment Mode",
    ["UPI", "Cash", "Card", "NetBanking"]
)

if st.sidebar.button("Add Expense"):
    new_data = pd.DataFrame(
        [[pd.to_datetime(date), category, amount, payment]],
        columns=["Date", "Category", "Amount", "Payment_Mode"]
    )

    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv("expenses.csv", index=False)

    st.sidebar.success("Expense Added Successfully")

# ---------------------------
# TOP METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Spending", f"Rs {data['Amount'].sum()}")
col2.metric("Average Daily Spend", f"Rs {round(data['Amount'].mean(), 2)}")
col3.metric("Total Transactions", len(data))

# ---------------------------
# CATEGORY CALCULATION
# ---------------------------
category_sum = data.groupby('Category')['Amount'].sum()

# ---------------------------
# DAILY TREND
# ---------------------------
st.subheader("Daily Spending Trend")

daily = data.groupby(data['Date'].dt.date)['Amount'].sum()
st.line_chart(daily)

# ---------------------------
# CATEGORY BAR CHART
# ---------------------------
st.subheader("Category-wise Spending")
st.bar_chart(category_sum)

# ---------------------------
# PIE CHART
# ---------------------------
st.subheader("Spending Distribution")

st.pyplot(category_sum.plot.pie(autopct='%1.1f%%').figure)

# ---------------------------
# MONTHLY PREDICTION
# ---------------------------
st.subheader("Monthly Expense Prediction")

data['Month'] = data['Date'].dt.month
monthly = data.groupby('Month')['Amount'].sum().reset_index()

if len(monthly) > 1:
    X = monthly[['Month']]
    y = monthly['Amount']

    model = LinearRegression()
    model.fit(X, y)

    next_month = [[monthly['Month'].max() + 1]]
    prediction = model.predict(next_month)

    st.success(f"Predicted Next Month Expense: Rs {round(prediction[0], 2)}")
else:
    st.warning("Not enough data for prediction")

# ---------------------------
# ALERT SYSTEM
# ---------------------------
st.subheader("Alerts")

for cat, amt in category_sum.items():
    if amt > 2000:
        st.warning(f"Overspending on {cat}: Rs {amt}")

# ---------------------------
# DATA TABLE
# ---------------------------
st.subheader("All Expenses")
st.dataframe(data)