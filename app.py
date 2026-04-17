import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Expense Tracker", layout="wide")

# ---------------------------
# DARK UI STYLE
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00ffcc;
}
</style>
""", unsafe_allow_html=True)

st.title("Smart Expense Tracker Dashboard")

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("expenses.csv")
data['Date'] = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
data = data.dropna(subset=['Date'])

# ---------------------------
# FILTERS
# ---------------------------
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start Date", data['Date'].min())
end_date = st.sidebar.date_input("End Date", data['Date'].max())

category_filter = st.sidebar.multiselect(
    "Select Category",
    options=data['Category'].unique(),
    default=data['Category'].unique()
)

filtered_data = data[
    (data['Date'] >= pd.to_datetime(start_date)) &
    (data['Date'] <= pd.to_datetime(end_date)) &
    (data['Category'].isin(category_filter))
]

# ---------------------------
# ADD EXPENSE
# ---------------------------
st.sidebar.header("Add Expense")

date = st.sidebar.date_input("Date", value=pd.to_datetime("today"))
category = st.sidebar.selectbox("Category", ["Food", "Travel", "Shopping", "Bills", "Entertainment"])
amount = st.sidebar.number_input("Amount", min_value=0)
payment = st.sidebar.selectbox("Payment Mode", ["UPI", "Cash", "Card", "NetBanking"])

if st.sidebar.button("Add Expense"):
    new_data = pd.DataFrame(
        [[pd.to_datetime(date), category, amount, payment]],
        columns=["Date", "Category", "Amount", "Payment_Mode"]
    )
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv("expenses.csv", index=False)
    st.sidebar.success("Expense Added")

# ---------------------------
# METRICS
# ---------------------------
total = filtered_data['Amount'].sum()
avg = filtered_data['Amount'].mean()
count = len(filtered_data)

col1, col2, col3 = st.columns(3)
col1.metric("Total Spending", f"Rs {total}")
col2.metric("Average Spend", f"Rs {round(avg, 2)}")
col3.metric("Transactions", count)

# ---------------------------
# TOP CATEGORY
# ---------------------------
category_sum = filtered_data.groupby('Category')['Amount'].sum()
top_category = category_sum.idxmax()

st.subheader(f"Top Spending Category: {top_category}")

# ---------------------------
# DAILY TREND
# ---------------------------
st.subheader("Daily Spending Trend")

daily = filtered_data.groupby(filtered_data['Date'].dt.date)['Amount'].sum()
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

fig, ax = plt.subplots()
category_sum.plot.pie(autopct='%1.1f%%', ax=ax)
st.pyplot(fig)

# ---------------------------
# MONTHLY SUMMARY
# ---------------------------
st.subheader("Monthly Summary")

filtered_data['Month'] = filtered_data['Date'].dt.to_period('M')
monthly = filtered_data.groupby('Month')['Amount'].sum()
st.bar_chart(monthly)

# ---------------------------
# ML PREDICTION
# ---------------------------
st.subheader("Expense Prediction")

monthly_df = monthly.reset_index()
monthly_df['Month'] = range(1, len(monthly_df)+1)

if len(monthly_df) > 1:
    X = monthly_df[['Month']]
    y = monthly_df['Amount']

    model = LinearRegression()
    model.fit(X, y)

    next_month = [[len(monthly_df) + 1]]
    pred = model.predict(next_month)

    st.success(f"Estimated Next Month Expense: Rs {round(pred[0], 2)}")

# ---------------------------
# BUDGET ALERT
# ---------------------------
st.subheader("Budget Check")

budget = st.number_input("Set Monthly Budget", min_value=0)

if budget > 0:
    if total > budget:
        st.error("You have exceeded your budget!")
    else:
        st.success("You are within budget")

# ---------------------------
# DATA TABLE
# ---------------------------
st.subheader("All Expenses")
st.dataframe(filtered_data)