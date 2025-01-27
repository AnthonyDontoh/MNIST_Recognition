from tensorflow.keras.utils import to_categorical

def perform_feature_engineering(X_train, X_test, y_train, y_test):
    print("Reshaping feature matrix to fit CNN model... this is done because we need to add a channel dimension to the data...")
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    print("One Hot Encoding the target variable...")
    y_cat_train = to_categorical(y_train, 10)
    y_cat_test = to_categorical(y_test, 10)

    print("Feature Scaling...")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255 # This is done to normalize the data to have akin values between 0 and 1
    X_test /= 255

    return X_train, X_test, y_cat_train, y_cat_test

    