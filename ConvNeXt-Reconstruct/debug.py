import torch

from skimage.io import imread
from skimage import img_as_ubyte
import os
from preprocessing.normalize import preprocess_signature
import numpy as np
from PIL import Image

canvas_size = (952, 1360)  # Maximum signature size

# If GPU is available, use it:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

# Load and pre-process the signature
original = img_as_ubyte(imread('/home/Work/signatureDatasets/PreTrain/左艳云.png', as_gray=True))
processed = preprocess_signature(original, canvas_size)

output_dir = '/home/Work/signatureDatasets/processed'
save_path = os.path.join(output_dir)

print(processed.dtype)
if processed.dtype != np.uint8:
    processed = (processed * 255).clip(0, 255).astype(np.uint8)

img = Image.fromarray(processed)
img.save(save_path)
print(f"✅ Saved processed image to: {save_path}")