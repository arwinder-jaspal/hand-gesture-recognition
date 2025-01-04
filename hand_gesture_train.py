from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Create a Sequential model
model = Sequential()

# Add convolutional layers to the model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
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
model.add(Dense(256, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('HandGestureDataset/train',
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    classes=['0', '1', '2', '3', '4', '5'])
validation_generator = validation_datagen.flow_from_directory('HandGestureDataset/test',
                                                              target_size=(256, 256),
                                                              batch_size=32,
                                                              class_mode='categorical',
                                                              classes=['0', '1', '2', '3', '4', '5'])
