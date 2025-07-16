import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 0: Expose the known sample for testing
def get_prediction_sample():
    """
    TODO: Return a 2D NumPy array for a known Setosa input sample.
    Example: [[5.1, 3.5, 1.4, 0.2]]
    """
    pass


# Step 1: Load and preprocess dataset
def load_and_preprocess():
    """
    TODO:
    - Load the Iris dataset using sklearn
    - Convert the target to binary: 1 if Setosa, else 0
    - Split the data into 80% train and 20% test sets
    - Scale the features using StandardScaler (fit on train, transform both)
    - Return: X_train, X_test, y_train, y_test, scaler
    """
    pass


# Step 2: Build model
def build_model():
    """
    TODO:
    - Create a tf.keras.Sequential model
    - Add a Dense layer with 10 units and 'relu' activation, input_shape=(4,)
    - Add an output Dense layer with 1 unit and 'sigmoid' activation
    - Compile the model using:
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    - Return the compiled model
    """
    pass


# Step 3: Train model
def train_model(model, X_train, y_train):
    """
    TODO:
    - Train the model using model.fit()
    - Set epochs=50, batch_size=8, verbose=0
    - Return the trained model
    """
    pass


# Step 4: Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    TODO:
    - Use model.evaluate() to get accuracy on test data
    - Print the accuracy using: Test Accuracy: 0.94 (4 decimal places)
    """
    pass


# Step 5: Predict a known Setosa sample
def predict_sample(model, scaler):
    """
    TODO:
    - Load the sample using get_prediction_sample()
    - Scale the sample using the given scaler
    - Use model.predict() to get the predicted probability
    - Print the prediction using:
        Predicted probability of being Setosa: 0.98 (4 decimal places)
    """
    pass


# Step 6: Main function to orchestrate training & evaluation
def main():
    """
    TODO:
    - Load and preprocess the data
    - Build the model
    - Train the model
    - Evaluate the model
    - Predict on a Setosa sample
    """
    pass


# Entry point
if __name__ == "__main__":
    main()
