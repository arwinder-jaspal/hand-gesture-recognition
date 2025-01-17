from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.callbacks import EarlyStopping, ModelCheckpoint

# Create a Sequential model
model = Sequential()

# Add convolutional layers to the model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())


# Add fully connected layers to the model
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 12.,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range=0.15,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('HandGestureDataset/train',
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    color_mode='grayscale',
                                                    class_mode='categorical',
                                                    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'])

validation_generator = validation_datagen.flow_from_directory('HandGestureDataset/validation',
                                                              target_size=(256, 256),
                                                              batch_size=32,
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'])

callback_list = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath='model.keras', monitor='val_loss', save_best_only=True, verbose=1)]

model.fit(train_generator,
          steps_per_epoch= 17,
          epochs=15,
          validation_data=validation_generator,
          callbacks=callback_list)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")