import pandas as pd


import numpy as np
df=pd.read_csv(r"D:\Dataset\uber.csv")
print("Before cleaning:", df.shape)
df.dropna(inplace=True)
df = df[df['fare_amount'] > 0]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
df['pickup_datetime'] 
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['pickup_datetime'] 
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
def distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

df['distance_km'] = distance(df['pickup_latitude'],
                             df['pickup_longitude'],
                             df['dropoff_latitude'],
                             df['dropoff_longitude'])
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
print("Before outlier removal:", df.shape)

# Remove very large fares and distances
df = df[(df['fare_amount'] < 100) & (df['distance_km'] < 50)]

print("After outlier removal:", df.shape)

# You can also visualize with boxplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['fare_amount'])
plt.title("Boxplot for Fare Amount (detecting outliers)")
plt.show()
corr = df[['fare_amount', 'distance_km', 'passenger_count', 'hour', 'day', 'month']].corr()
print("\nCorrelation with fare_amount:")
print(corr['fare_amount'])

# Visual heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
X = df[['distance_km', 'passenger_count', 'hour']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=101)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
def evaluate(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} → R²: {r2:.3f} | RMSE: {rmse:.3f}")

print("\nModel Evaluation Results:")
evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")
