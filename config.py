class CONFIG:
    
    IMG_PATH = '../input/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax/png_images'
    MASK_PATH = '../input/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax/png_masks'
    train_csv = '../input/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax/stage_1_train_images.csv'
    test_csv = '../input/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax/stage_1_test_images.csv'

    BATCH_SIZE= 16
    IMG_CHANNEL_IMAGE = 3
    IMG_CHANNEL_MASK = 1
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNEL = 3
    EPOCHS =10
    train_test_split =0.1
