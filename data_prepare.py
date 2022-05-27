import numpy as np
from config import CONFIG
import tqdm
from tqdm import tqdm
import os

def preprocess_image_labels(data_with_pneumo):
    "Function to preprocess train and test data"
    
    print("Getting and Resizing Images and Mask ...\n\n")

    # initialise matrices to zeros with respective dimensions
    X_seg = np.zeros((len(data_with_pneumo), CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, CONFIG.IMG_CHANNEL_IMAGE), dtype=np.uint8)
    Y_seg = np.zeros((len(data_with_pneumo), CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, CONFIG.IMG_CHANNEL_MASK), dtype=np.bool)

    img_data = list(data_with_pneumo.T.to_dict().values())

    #Create matrices for images,label and mask
    for i, data_row in tqdm(enumerate(img_data), total=len(img_data)):

        patientImage = data_row['new_filename']
        imageLabel  = data_row['has_pneumo']

        imagePath = os.path.join(CONFIG.IMG_PATH, patientImage)
        lungImage = cv2.imread(imagePath)
        lungImage = cv2.resize(lungImage, (CONFIG.IMG_WIDTH,CONFIG.IMG_HEIGHT))

        X_seg[i] = lungImage

        maskPath = os.path.join(CONFIG.MASK_PATH, patientImage)
        maskImage = cv2.imread(maskPath,0)
        maskImage = cv2.resize(maskImage, (CONFIG.IMG_WIDTH,CONFIG.IMG_HEIGHT))
        maskImage = np.expand_dims( maskImage,axis=-1)

        Y_seg[i] = maskImage

    print('\n\nProcess ... C O M P L E T E')

    return X_seg,Y_seg
