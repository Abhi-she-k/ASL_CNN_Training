import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
                                     ZeroPadding2D, GlobalAveragePooling2D, Dense,
                                     Dropout, SpatialDropout2D, MaxPooling2D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from datetime import datetime

def compute_info():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def create_model_subsets(file_path, class_names, img_size, validation_split, subset, batch_size):

    preprocessed_data = tf.keras.preprocessing.image_dataset_from_directory(
        directory = file_path, 
        labels='inferred',
        validation_split = validation_split,
        label_mode='int',
        subset = subset,
        seed=123,
        class_names= class_names,    
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        verbose=True
    )

    return preprocessed_data


def normalize_img(tensorData):
    normalization_layer = layers.Rescaling(1./255)
    normalized_data = tensorData.map(lambda x, y: (normalization_layer(x), y))

    return normalized_data

def test_display(data_batch):
    
    for images, labels in data_batch.take(1):
    
        plt.figure(figsize=(10, 10))

        for i in range(16):
            plt.subplot(16, 16, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow((images[i].numpy()* 255).astype("uint8"))  # squeeze removes (256, 256, 1) -> (256, 256)
            plt.xlabel(class_names[labels[i].numpy()])
        
    plt.show()

def model():
    
    model = Sequential()
    
    return model


def model_layer(model, filter, pooling_stride, padding, activation, input_shape, dropout):

    if(input_shape):

        model.add(Conv2D(filter, (3,3), strides=(1,1), padding=padding, input_shape=input_shape))  
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=pooling_stride))

        return model
    

    else:

        model.add(Conv2D(filter, (3,3), strides=(1,1), padding=padding))   
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=pooling_stride))
        model.add(Dropout(dropout))

        return model


def flatten(model):

    model.add(Flatten())

    return model
    
def dense_layer(model, filter, classification, dropout, activation, final_layer):

    if(final_layer):
        
        model.add(Dense(classification))
        model.add(Activation(activation))

        return model
        
    else:
        
        model.add(Dense(filter))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    
        return model
    
def model_settings(model, learning_rate):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


if __name__ == "__main__":

    print("Starting Model...\n")
    
    compute_info()

    file_path = "/kaggle/input/2025-08-18/newData2025-08-18"

    # SETTINGS
    input_shape = (256,256,3)
    epochs = 20
    loss_factor = 0.01
    img_size = (256,256)
    batch_size = 32
    val_split = 0.2
    learning_rate = 0.001

    class_names = ["A", "B", "C", "D", "E", "F", "G", "I", "J", "K", "L", "M", "N", "Nothing", "O", "P", "Q", "R"]

    classification = len(class_names)

    now = datetime.now()
    time_now = now.strftime("%H-%M-%S")

    training_data = create_model_subsets(file_path+"/training", class_names, img_size, val_split, "training", batch_size)

    val_data = create_model_subsets(file_path+"/training", class_names, img_size, val_split, "validation", batch_size)

    test_data = create_model_subsets(file_path+"/test", class_names, img_size, 0, None, batch_size)

    
    model = model()
    
    model_layer(model, 32, (2,2), 'same', 'relu', input_shape, 0)
    model_layer(model, 64, (2,2), 'same', 'relu', False, 0)
    model_layer(model, 128, (2,2), 'same', 'relu', False,  0)
    model_layer(model, 256, (2,2), 'same', 'relu', False,  0)
    model_layer(model, 512, (2,2), 'same', 'relu', False,  0.6)
    model_layer(model, 1024, (4,4), 'same', 'relu', False,  0.6)

    flatten(model)

    dense_layer(model, 256, 0, 0.3, 'relu', False)
    dense_layer(model, 128, 0, 0.2, 'relu', False)
    dense_layer(model, 64, 0, 0.15, 'relu', False)
    dense_layer(model, 0, classification, 0, 'softmax', True)

    model.summary()

    model_settings(model, learning_rate)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=loss_factor, patience=3)
    ]

    history = model.fit(
        training_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.2])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([1.0e-12, 10])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_data, verbose=2)

    print(f"Test Accuracy: {test_acc}")


    model.save(f"ASLCustom{time_now}.keras")
