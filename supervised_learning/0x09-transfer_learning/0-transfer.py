#!/usr/bin/env python3
""" transfer"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ fix data"""
    x = K.applications.resnet50.preprocess_input(X)
    y = K.utils.to_categorical(Y)
    return x, y


if __name__ == "__main__":

    def resize_images(X):
        """resize data so it is accepted by ResNet50 model"""
        return K.backend.resize_images(X, 7, 7,
                                       data_format="channels_last")
    (X_train, Y_train), (
     X_valid, Y_valid) = K.datasets.cifar10.load_data()
    xtrain, ytrain = preprocess_data(X_train, Y_train)
    xtest, ytest = preprocess_data(X_valid, Y_valid)

    input = K.Input(shape=(32, 32, 3))
    res_model = K.applications.ResNet50(weights='imagenet',
                                        include_top=False,
                                        input_shape=(224, 224, 3))

    for layer in res_model.layers[:143]:
        layer.trainable = False
    model = K.models.Sequential()
    model.add(K.layers.Lambda(resize_images))
    model.add(res_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    calls = [K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                         save_best_only=True)]

    opt = K.optimizers.RMSprop(lr=2e-5)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(xtrain, ytrain,
              batch_size=32,
              epochs=10,
              validation_data=(xtest, ytest),
              shuffle=True,
              callbacks=calls)
