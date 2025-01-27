import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from ML_Pipeline import feature_engineering
from ML_Pipeline import model

from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.simplefilter(action='ignore')

try:
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # we use tuple unpacking to load the data. 
    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_test: ", y_test.shape)

    # Perform Feature Engineering:
    print("Performing Feature Engineering...")
    X_train, X_test, y_cat_train, y_cat_test = feature_engineering.perform_feature_engineering(X_train, X_test, y_train, y_test)

    # Build the model:
    print("CNN Model Building...")
    cnn_model = model.create_cnn_model()

    # Train and saving model:
    print("Training and saving the CNN model...")
    trained_cnn_model = model.train_and_save_model(cnn_model, X_train, y_cat_train, X_test, y_cat_test)

    # Model Performance during training and validation:
    print("Model Performance during training and validation...")
    training_metrics = pd.DataFrame(trained_cnn_model.history.history)
    print(training_metrics.columns)
    print("Training Metrics: \n", training_metrics.columns)

    # Model Evaluation:
    score = trained_cnn_model.evaluate(X_test, y_cat_test, verbose=0)
    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])

    # Model Prediction:
    print("Model Prediction...")
    predictions = np.argmax(trained_cnn_model.predict(X_test), axis=-1)
    print("Classification Report: \n", classification_report(y_test, predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))

    # Save the predictions:
    prediction_df = pd.DataFrame({'Label': predictions}) # this dataframe will have a single column called 'Label' which contains the predictions
    path = r"C:\Users\adontoh\Desktop\MNIST_Recognition\Output\Predictions.csv"
    prediction_df.to_csv(path, index=False) # index is set to False to avoid saving the index column

except Exception as e:
    print("An error occurred: ", e)


