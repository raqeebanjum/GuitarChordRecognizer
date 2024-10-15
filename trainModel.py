import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define the dataset directory
dataset_dir = 'trainingData/'  # Path to your dataset folder

# Update the image size to a larger resolution
img_size = (512, 512)
batch_size = 32

# Load dataset with the new image size
train_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,  # Resize the images to 512x512
    batch_size=batch_size,
    label_mode='categorical',  # Categorical for class labels
    shuffle=True
)

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

# Build the CNN model (same as before, just with updated input shape)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),  # Input shape updated
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes: emptyFrets, gMajor
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the new image size
epochs = 10
history = model.fit(normalized_train_dataset, epochs=epochs)

# Save the model
model.save('model/guitar_chord_model_512.keras')