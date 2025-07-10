import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Step 1: Load and preprocess dataset
def load_and_preprocess():
    # ✅ Load Iris dataset using scikit-learn
    iris = load_iris()
    X = iris.data  # Shape: (150, 4)

    # ✅ Convert labels to binary: 1 if Setosa, else 0
    y = (iris.target == 0).astype(int)

    # ✅ Split into 80% train and 20% test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Scale features to standard normal distribution
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ Return processed data and scaler for future use
    return X_train, X_test, y_train, y_test, scaler


# Step 2: Build the model
def build_model():
    # TODO: Create a Sequential model
    #       Add an input layer that accepts 4 features (input_shape=(4,))
    #       Add a Dense hidden layer with 10 units and ReLU activation
    #       Add an output Dense layer with 1 unit and sigmoid activation
    #       Compile the model using:
    #         - Optimizer: Adam
    #         - Loss: binary_crossentropy
    #         - Metrics: accuracy
    #       Return the compiled model
    pass


# Step 3: Train the model
def train_model(model, X_train, y_train):
    # TODO: Train the model using model.fit() with the following settings:
    #         - Inputs: X_train, y_train
    #         - Epochs: 50
    #         - Batch size: 8
    #         - Verbose: 0 (to suppress output)
    #       Return the trained model
    pass


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    # TODO: Use model.evaluate() on test data
    #       Extract and print the test accuracy in format:
    #         Test Accuracy: <rounded_value>
    pass


# Step 5: Predict a sample
def predict_sample(model, scaler):
    # TODO: Create a NumPy array for a sample Setosa input:
    #         Example: [[5.1, 3.5, 1.4, 0.2]]
    #       Scale the input using the scaler
    #       Use model.predict() to get prediction
    #       Print predicted probability in format:
    #         Predicted probability of being Setosa: <rounded_value>
    pass


# Step 6: Main function
def main():
    # ✅ Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    # TODO: Call build_model() to get the model
    # TODO: Train the model with train_model()
    # TODO: Evaluate the model with evaluate_model()
    # TODO: Predict on one sample with predict_sample()
    pass


# Entry point
if __name__ == "__main__":
    main()
