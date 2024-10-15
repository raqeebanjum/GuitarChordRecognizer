import os
import cv2
import mediapipe as mp

# Initialize MediaPipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define the directories
input_dir = 'trainingData/gMajor/'  # Folder containing your images
output_dir = 'croppedTrainingData/gMajor/'  # Folder where cropped images will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to detect hands and crop around them
def crop_hand(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # Get bounding box of the hand
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = image.shape
        min_x, min_y = w, h
        max_x, max_y = 0, 0

        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)

        # Add some padding around the hand
        padding = 20
        min_x = max(min_x - padding, 0)
        min_y = max(min_y - padding, 0)
        max_x = min(max_x + padding, w)
        max_y = min(max_y + padding, h)

        # Crop the hand region
        cropped_img = image[min_y:max_y, min_x:max_x]
        return cropped_img
    else:
        # If no hand is detected, return the original image
        return image

# Loop over all images in the input directory
# Loop over all images in the input directory
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    # Ensure that it’s a file (not a directory)
    if os.path.isfile(img_path):
        # Read the image
        img = cv2.imread(img_path)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Error: Unable to load image at {img_path}")
            continue  # Skip this image and move to the next

        # Detect hand and crop the image
        cropped_img = crop_hand(img)

        # Save the cropped image in the output folder
        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, cropped_img)

print("Cropping complete!")