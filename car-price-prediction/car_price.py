import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load the car dataset
df = pd.read_csv("car_data.csv")
print(df.head())

# Features and target
X = df[['Engine_Size', 'Car_Age']]
y = df['Selling_Price']

# Train a Decision Tree Regression model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Predict price for a new car
new_car = pd.DataFrame([[1300, 3]], columns=['Engine_Size', 'Car_Age'])
prediction = model.predict(new_car)
print("Predicted Selling Price:", round(prediction[0], 2))
