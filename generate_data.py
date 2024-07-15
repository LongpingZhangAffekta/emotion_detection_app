from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class GenerateData:
    def __init__(self, data_path):
        # Initialize the class with the path to the data folder.
        self.data_path = data_path

    def split_test(self, val_filename='val'):
        # Splits the validation and train data from the general train data file.
        train_csv_path = os.path.join(self.data_path, 'train.csv')  # Construct the path to the train CSV file.
        train_data = pd.read_csv(train_csv_path)  # Read the train data from the CSV file.

        # Split the data into validation and train sets.
        validation_data = train_data.iloc[:3589, :]
        train_data = train_data.iloc[3589:, :]

        # Save the split data back to CSV files.
        train_data.to_csv(os.path.join(self.data_path, 'train.csv'), index=False)
        validation_data.to_csv(os.path.join(self.data_path, f"{val_filename}.csv"), index=False)

        print("Done splitting the test file into validation & final test file")

    def str_to_image(self, image_str=' '):
        # Converts a string of pixel values to a PIL Image object.
        image_array = np.asarray(image_str.split(' '), dtype=np.uint8).reshape(48, 48)  # Convert the string to a numpy array and reshape.
        return Image.fromarray(image_array)  # Convert the numpy array to a PIL Image object.

    def save_images(self, data_type='train'):
        # Saves images from a CSV data file to a specified folder.
        folder_name = os.path.join(self.data_path, data_type)  # Construct the path to the folder where images will be saved.
        csv_file_path = os.path.join(self.data_path, f"{data_type}.csv")  # Construct the path to the CSV file.

        # Create the folder if it does not exist.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        data = pd.read_csv(csv_file_path)  # Read the data from the CSV file.
        images = data['pixels']  # Extract the pixel values.
        num_images = len(images)  # Get the number of images.

        # Iterate over each image and save it to the specified folder.
        for index in tqdm(range(num_images), desc=f"Saving {data_type} images"):
            img = self.str_to_image(images.iloc[index])  # Convert the string of pixels to an image.
            img.save(os.path.join(folder_name, f"{data_type}{index}.jpg"), 'JPEG')  # Save the image to the folder.

        print(f"Done saving {data_type} data")


# Example usage:
# data_generator = GenerateData('path/to/data')
# data_generator.split_test()
# data_generator.save_images('train')
