import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# Fixing ₹ symbol issue
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set plot style
sns.set(style="whitegrid")

# Required columns
required_columns = ['Date', 'Total Sales', 'Net Revenue', 'Product', 'Units', 'Region', 
                    'City', 'Shipping Fee', 'Order Status', 'Discount (%)', 'Payment Method', 'Day']

# Load the Excel file
file_path = os.path.join(os.getcwd(), "Data set.xlsx")
df = pd.read_excel(file_path, sheet_name="DATA")

# Show column names
print("\nColumn names:\n", df.columns.tolist())

# Validate required columns
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise KeyError(f"Missing required columns: {missing_columns}")

# Check for null values
print("\nNull values in each column:\n", df.isnull().sum())

# Handle missing values
df = df.dropna(subset=['Date', 'Net Revenue', 'Units'])

# Convert date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Create Day column
df['Day'] = df['Date'].dt.day_name()

# SIMPLE LINEAR REGRESSION

df_reg = df[df['Order Status'] != 'Cancelled']

X = df_reg[['Total Sales']]
y = df_reg['Net Revenue']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\n--- Simple Linear Regression ---")
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

# Plot regression
plt.figure(figsize=(8,5))
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel("Total Sales")
plt.ylabel("Net Revenue")
plt.title("Linear Regression: Total Sales vs Net Revenue")
plt.tight_layout()
plt.savefig("linear_regression.png")
plt.show()

# EDA

# 1. Daily Sales
daily_sales = df.groupby('Date')[['Total Sales', 'Net Revenue']].sum().sort_index()
daily_sales.plot(figsize=(12, 6), title="Daily Total Sales vs Net Revenue")
plt.ylabel("Amount (Rs)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("daily_sales.png")
plt.show()

# 2. Top Products (Units)
top_products = df.groupby('Product')['Units'].sum().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index, hue=top_products.index, legend=False)
plt.title("Top 10 Best-Selling Products (Units)")
plt.xlabel("Units Sold")
plt.tight_layout()
plt.savefig("top_products_units.png")
plt.show()

# 3. Top Products (Revenue)
top_revenue_products = df.groupby('Product')['Net Revenue'].sum().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index, hue=top_revenue_products.index, legend=False)
plt.title("Top 10 Products by Net Revenue")
plt.xlabel("Revenue (Rs)")
plt.tight_layout()
plt.savefig("top_products_revenue.png")
plt.show()

# 4. Revenue by Region
region_sales = df.groupby('Region')['Net Revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=region_sales.index, y=region_sales.values, hue=region_sales.index, legend=False)
plt.title("Net Revenue by Region")
plt.ylabel("Revenue (Rs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("region_sales.png")
plt.show()

# 5. City Shipping vs Revenue
city_group = df.groupby('City')[['Shipping Fee', 'Net Revenue']].sum().nlargest(10, 'Net Revenue')
city_group.plot(kind='bar', figsize=(12, 6), title="Top Cities: Shipping Fee vs Net Revenue")
plt.ylabel("Amount (Rs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("city_shipping_revenue.png")
plt.show()

# 6. Order Status
plt.figure(figsize=(6, 6))
df['Order Status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Order Status Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("order_status.png")
plt.show()

# 7. Units Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Units'], bins=30, kde=True)
plt.title("Units per Transaction")
plt.xlabel("Units")
plt.tight_layout()
plt.savefig("units_per_transaction.png")
plt.show()

# 8. Payment Method
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Payment Method', order=df['Payment Method'].value_counts().index,
              hue='Payment Method', legend=False)
plt.title("Payment Method Preference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("payment_methods.png")
plt.show()

# 9. Day-wise Revenue
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales = df.groupby('Day')['Net Revenue'].sum().reindex(day_order)

plt.figure(figsize=(10, 6))
sns.barplot(x=day_sales.index, y=day_sales.values, hue=day_sales.index, legend=False)
plt.title("Net Revenue by Day")
plt.ylabel("Revenue (Rs)")
plt.tight_layout()
plt.savefig("day_sales.png")
plt.show()

plt.close('all')
