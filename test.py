import numpy as np
import pandas as pd
from config import CONFIG
from data_prepare import preprocess_image_labels
import matplotlib.pyplot as plt
from train import create_train_model

#get 3 test samples preprocessed
test_data = pd.read_csv(CONFIG.test_csv)
test_data_pneumo = test_data[test_data['has_pneumo'] == 1]
test_data_pneumo = test_data_pneumo.iloc[2:5]
index = pd.Index(range(1, 4, 1))
test_data_pneumo = test_data_pneumo.set_index(index)
X_test_seg, Y_test_seg= preprocess_image_labels( data_with_pneumo = test_data_pneumo )

#prediction
model = create_train_model()
pred = model.predict(X_test_seg)


#plot prediction and ground truth of test results
fig = plt.figure(figsize=(10, 7))
for i in range(3):
    fig.add_subplot(3, 3, (i*3)+1)
    plt.imshow(X_test_seg[i])
    plt.title("image")
    fig.add_subplot(3, 3, (i*3)+2)
    plt.imshow(Y_test_seg[i])
    plt.title("truth")
    fig.add_subplot(3, 3,(i*3)+3)
    plt.imshow(pred[i])
    plt.title("pred")
