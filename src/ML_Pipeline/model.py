from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def create_cnn_model():
    input_shape = (28, 28, 1)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model Summary: \n", model.summary())
    return model

def train_and_save_model(cnn_model, X_train, y_cat_train, X_test, y_cat_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    cnn_model.fit(X_train, y_cat_train, epochs=50, validation_data=(X_test, y_cat_test), callbacks=[early_stop])
    print("Model Training Completed...")
    cnn_model.save(r"C:\Users\adontoh\Desktop\MNIST_Recognition\Output\mnist_cnn_model.keras") #saving the model for future use
    print("Saving the model as mnist.h5")
    return cnn_model
