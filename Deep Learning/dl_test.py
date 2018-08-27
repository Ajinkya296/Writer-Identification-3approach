import numpy as np
import pandas as pd
import cv2 as cv
from keras.models import Sequential, Model,load_model,model_from_json
def norm_resize(img):
    img1 = img / 255
    img = cv.resize(img1,(160,70))
    return img
testimages_path  =    'Hidden Test Folder/DLHiddenTest/DLTestData/'
pairs_list =    np.genfromtxt('Hidden Test Folder/DLHiddenTest/DLTestPairs.csv',delimiter=',')
pairs_df   =    pd.read_csv('Hidden Test Folder/DLHiddenTest/DLTestPairs.csv')

test_pairs = []
for i in range(pairs_df.shape[0]):
    img1 = norm_resize(cv.imread(testimages_path + str(pairs_df['FirstImage'][i]),0))
    img2 = norm_resize(cv.imread(testimages_path + str(pairs_df['SecondImage'][i]),0))
    test_pairs +=[[img1,img2]]
test_pairs = np.array(test_pairs).reshape((-1,2,70,160,1))
print(test_pairs[:,0,:,:,:].shape)

with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model_weights.h5')
print('MODEL LOADED ')
pred = model.predict([test_pairs[:,0,:,:,:],test_pairs[:,1,:,:,:]],batch_size=128)
rounded = [round(-x) + 1 for x in pred]
pairs_df['SameOrDifferent'] = rounded
print(pairs_df)
pairs_df.to_csv('Hidden Test Folder/DLHiddenTest/DLTestOutput2.csv',sep=',')
