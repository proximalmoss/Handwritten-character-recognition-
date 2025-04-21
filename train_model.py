import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data-
data = pd.read_csv("A_Z Handwritten Data.csv").astype("float32")
x = data.drop("0", axis=1).values
y = data["0"].values

# Reshape and normalize (no inversion)
x = x.reshape(-1, 28, 28, 1)
x = x / 255.0
y_cat = to_categorical(y, num_classes=26)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y_cat, test_size=0.2, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# Complex CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Training
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr, early_stop]
)

model.save('model_hand.h5')
print("Model saved as model_hand.h5")

# Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"📊 Model Evaluation on Test Data: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}")
print("The training accuracy is: ", history.history['accuracy'][-1])
print("The validation accuracy is: ", history.history['val_accuracy'][-1])
print("The training loss is: ", history.history['loss'][-1])
print("The validation loss is: ", history.history['val_loss'][-1])

# Plot accuracy and loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()