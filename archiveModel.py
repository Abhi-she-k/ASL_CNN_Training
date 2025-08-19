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