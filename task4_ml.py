import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Sample dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,40,50,55,60,65,70,75,80,90]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours"]]   # Input
y = df["Marks"]     # Output


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
hours = float(input("Enter study hours: "))
prediction = model.predict([[hours]])

print("Predicted Marks:", round(prediction[0],2))


# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X), linestyle="--")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
