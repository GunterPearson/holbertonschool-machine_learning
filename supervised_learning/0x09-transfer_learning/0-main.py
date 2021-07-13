import tensorflow.keras as K


def preprocess_data(X, Y):
    """ fix data"""
    x = K.applications.resnet50.preprocess_input(X)
    y = K.utils.to_categorical(Y)
    return x, y
    


if __name__ == "__main__":

    def resize_images1(X):
        """resize data so it is accepted by ResNet50 model"""
        return K.backend.resize_images(X, 7, 7,
                                       data_format="channels_last")
    (X_train, Y_train), (
     X_valid, Y_valid) = K.datasets.cifar10.load_data()

    input = K.Input(shape=(32, 32, 3))
    lam = K.layers.Lambda(resize_images1)(input)
    # reg = K.regularizers.l2(0.001)
    RES = K.applications.ResNet50(weights='imagenet',
                                  include_top=False,
                                  input_shape=(224, 224, 3))
    RES.trainable = False
    res = RES(lam)

    flat = K.layers.Flatten()(res)
    batch = K.layers.BatchNormalization()(flat)
    dense = K.layers.Dense(1000, activation='relu', kernel_initializer='he_normal')(batch)
    drop = K.layers.Dropout(.3)(dense)
    batch1 = K.layers.BatchNormalization()(drop)
    # kernel_regularizer=reg
    # dense = K.layers.Dense(1000, activation='relu',
    #                        kernel_initializer='he_normal')(batch)
    dense1 = K.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(batch1)
    drop1 = K.layers.Dropout(.2)(dense1)
    batch2 = K.layers.BatchNormalization()(drop1)
    dense2 = K.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(batch2)
    drop2 = K.layers.Dropout(.3)(dense2)
    batch3 = K.layers.BatchNormalization()(drop2)
    dense3 = K.layers.Dense(10, activation='softmax')(batch3)


    model = K.Model(input, dense3)
    opt = K.optimizers.SGD(lr=0.01,
                           momentum=0.9,
                           decay=1e-5)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 20
    batch_size = 64
    xtrain, ytrain = preprocess_data(X_train, Y_train)
    xtest, ytest = preprocess_data(X_valid, Y_valid)
    my_callbacks = [K.callbacks.ModelCheckpoint(
                    filepath='cifar10.h5', save_best_only=True)]
    model.fit(xtrain, ytrain,
              batch_size=batch_size,
              validation_data=(xtest, ytest),
              epochs=epochs,
              shuffle=True,
              callbacks=my_callbacks)
