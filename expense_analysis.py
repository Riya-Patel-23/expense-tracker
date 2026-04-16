import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("expenses.csv")
data['Date'] = pd.to_datetime(data['Date'])

print("Total Spending:", data['Amount'].sum())

category_sum = data.groupby('Category')['Amount'].sum()
print("\nCategory-wise Spending:\n", category_sum)

category_sum.plot(kind='bar')
plt.show()

daily = data.groupby('Date')['Amount'].sum()
daily.plot()
plt.show()

data['Day'] = data['Date'].dt.day

X = data[['Day']]
y = data['Amount']

model = LinearRegression()
model.fit(X, y)

future_days = pd.DataFrame({'Day': [31, 32, 33, 34, 35]})
predictions = model.predict(future_days)

print("\nPredictions:")
print(predictions)

# ---------------------------
# 📅 MONTHLY PREDICTION
# ---------------------------
data['Month'] = data['Date'].dt.month

monthly = data.groupby('Month')['Amount'].sum().reset_index()

X = monthly[['Month']]
y = monthly['Amount']

model = LinearRegression()
model.fit(X, y)

# Predict next month
next_month = [[monthly['Month'].max() + 1]]
prediction = model.predict(next_month)

print("\nPredicted Next Month Expense:", round(prediction[0], 2))