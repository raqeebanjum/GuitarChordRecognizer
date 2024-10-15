import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Defining directories
input_dir = 'trainingData/emptyFrets/'
output_dir = 'augmentedEmptyFrets/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize the ImageDataGenerator with augmentations
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment each image in the emptyFrets folder
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)

    # Load and convert the image to a NumPy array
    img = load_img(img_path)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to include batch dimension

    # Generate multiple augmented versions and save them
    i = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir,
                              save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 5:  # Save 5 augmented versions per image
            break  # Stop after generating 5 images per input image

print("Data augmentation complete!")