from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('model/guitar_chord_model_512.keras')

# Load a new image for prediction
img_path = '/Users/user/Desktop/GuitarChordRecognition/unseenData/4.JPG'
img = image.load_img(img_path, target_size=(512, 512))  # Resize to match model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize the image
img_array /= 255.0

# Predict the class
prediction = model.predict(img_array)

# Map predicted class to label
class_names = ['emptyFrets', 'gMajor']  # Assuming these are your class names
predicted_class = np.argmax(prediction)  # Get the index of the highest probability

# Convert the probabilities to percentages
probabilities = prediction[0] * 100  # Multiply by 100 to convert to percentages

# Print the results
print(f'Prediction: {class_names[predicted_class]}')
print(f'emptyFrets: {probabilities[0]:.2f}%')
print(f'gMajor: {probabilities[1]:.2f}%')