from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the built-in handwritten digits dataset
digits = load_digits()
X = digits.data
y = digits.target
print("Dataset shape:", X.shape)
print("Number of classes:", len(set(y)))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the MLP (neural network) classifier
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
