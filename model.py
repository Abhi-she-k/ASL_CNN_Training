import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
                                     ZeroPadding2D, GlobalAveragePooling2D, Dense,
                                     Dropout, SpatialDropout2D, MaxPooling2D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def compute_info():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def create_model_subsets(file_path, class_names, img_size=(256,256), validation_split=0, subset=None, batch_size=32):

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


# def normalize_img()







# normalization_layer = layers.Rescaling(1./255)
# trainData = trainData.map(lambda x, y: (normalization_layer(x), y))
# testData = testData.map(lambda x, y: (normalization_layer(x), y))
# valData = valData.map(lambda x, y: (normalization_layer(x), y))

# for images, labels in trainData.take(1):
#     plt.figure(figsize=(10, 10))
#     for i in range(16):
#         plt.subplot(16, 16, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow((images[i].numpy()* 255).astype("uint8"))  # squeeze removes (256, 256, 1) -> (256, 256)
#         plt.xlabel(class_names[labels[i].numpy()])
# plt.show()

# model = Sequential()

# model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=(256, 256, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Dropout(0.5))

# model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Dropout(0.5))

# model.add(Flatten())

# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.35))

# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.15))

# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.05))


# model.add(Dense(8))
# model.add(Activation('softmax'))

# # pretrainedModel = tf.keras.applications.MobileNetV2(
# #     input_shape=(256, 256, 3),
# #     include_top=False,
# #     alpha=1,
# #     weights='imagenet',
# #     pooling=None  
# # )

# # pretrainedModel.trainable = False

# # # for layer in pretrainedModel.layers[-9:]:
# # #     layer.trainable = True

# # inputs = pretrainedModel.input

# # x = tf.keras.layers.GlobalAveragePooling2D()(pretrainedModel.output)

# # # x = tf.keras.layers.Dense(512, activation='relu')(x)
# # # x = tf.keras.layers.Dropout(0.2)(x)

# # # x = tf.keras.layers.Dense(256, activation='relu')(x)
# # # x = tf.keras.layers.Dropout(0.2)(x)

# # x = tf.keras.layers.Dense(128, activation='relu')(x)

# # outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

# # model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.summary()

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# model.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=3)
# ]

# history = model.fit(
#     trainData,
#     epochs=20,
#     validation_data=valData,
#     callbacks=callbacks
# )


# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1.2])
# plt.legend(loc='lower right')
# plt.show()

# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label = 'val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim([1.0e-12, 10])
# plt.legend(loc='lower right')
# plt.show()

# test_loss, test_acc = model.evaluate(testData, verbose=2)

# print(test_acc)

# model.save("ASLCustomABCDEDFGBEST.keras")



if __name__ == "__main__":

    print("Starting Model...\n")
    
    compute_info()

    file_path = ""

    img_size = (256,256)
    batch_size = 32
    val_split = 0.2
    # class_names = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    #                "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"]
    class_names = ["A", "B", "C", "D", "E", "F", "G","nothing"]


    training_data = create_model_subsets(file_path+"/Training", class_names, img_size, val_split, "training", batch_size)

    val_data = create_model_subsets(file_path+"/Training", class_names, img_size, val_split, "validation", batch_size)

    training_data = create_model_subsets(file_path+"/Testing", class_names, img_size, batch_size)