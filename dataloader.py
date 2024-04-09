import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

class LabeledDataset(keras.utils.Sequence):
    """Helper to iterate over the LABELED data (as Numpy arrays)."""

    def __init__(self, input_path, mask_path, image_size=(224, 224), batch_size=4):
        self.input_path = input_path
        self.mask_path = mask_path
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        """Returns length of iteration by calculating total image / batch size"""
        return len(self.mask_path) // self.batch_size
    
    def __getitem__(self, index):
        """Returns tuple (input, mask) correspond to batch size #index"""
        i = index * self.batch_size

        batch_input_paths = self.input_path[i: i + self.batch_size]
        batch_mask_paths = self.mask_path[i: i + self.batch_size]

        x = np.zeros((self.batch_size, ) + self.image_size + (1,), dtype='float32')
        y = np.zeros((self.batch_size, ) + self.image_size + (1,), dtype='float32')

        for j, (image_path, mask_path) in enumerate(zip(batch_input_paths, batch_mask_paths)):
            image = load_img(image_path, target_size=self.image_size, color_mode='grayscale')
            mask = load_img(mask_path, target_size=self.image_size, color_mode="grayscale")

            x[j] = np.expand_dims(image, 2) / 255.0
            y[j] = np.round(np.expand_dims(mask, 2) / 255.0)
            
     
        return x, y


