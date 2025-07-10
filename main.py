import tensorflow as tf
import os

# Set the path to your training directory
train_dir = "train"
img_height = 180
img_width = 180
batch_size = 8

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),  # Resize images
    batch_size=batch_size
)

# Display the class names detected (should be ['cats', 'dogs'])
class_names = train_ds.class_names
print("Class names:", class_names)

# Show a few images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

print("Training the model now...")

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=5,
    verbose=1
)
print("👀 If you see this, the training code was reached")
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to any test image
img_path = "test/dogs/dog1.jpg"  # Change this to your actual test image

# Load and preprocess the image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
img_array = normalization_layer(img_array)

# Predict
prediction = model.predict(img_array)
score = prediction[0][0]

# Interpret result
if score > 0.5:
    print(f"This image is likely a: 🐶 Dog ({score:.2f})")
else:
    print(f"This image is likely a: 🐱 Cat ({1-score:.2f})")
model.save("model.h5")
