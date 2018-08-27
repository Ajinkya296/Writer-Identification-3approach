from keras.layers import Input, Conv2D, Lambda,Dropout, Dense, Flatten,MaxPooling2D,Activation
from keras.layers.normalization import BatchNormalization

import tensorflow as T
from keras.models import Model, Sequential,load_model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np
import os,random
import cv2 as cv
from keras import backend as K
from collections import Counter
from keras.utils import plot_model
import pandas as pd
path="../Simple Machine Learning/Images"    
K.set_epsilon(1e-04)

def dataset_init():
    #dataset initialization happens here
    writers=[]
    img_list=[]
    dataset=[[] for i in range(1556)]
    curr_writer=-1
    i=0
    for curr_img in os.listdir(path):
        img1 = cv.imread(os.path.join(path,curr_img),0)
        img = img1 / 255
        img=cv.resize(img,(160,70))
        img_list.append(img)
    for curr_img in os.listdir(path):
        writer_name = curr_img[:4]
        if writer_name not in writers:
            writers.append(writer_name)
            curr_writer+=1
        dataset[curr_writer].append(img_list[i])
        i+=1
    print("this is writer shape=", np.array(writers).shape)
    return np.array(dataset),writers

def siamese_euclidean_distance(pair):
#calculate euclidean distance
    u,v = pair[0],pair[1]
    diff = T.subtract(u,v)
    sqr  = T.pow(diff,2)
    sum_=T.reduce_sum(sqr,1,keep_dims=True)
    dist=T.sqrt(sum_)
    return dist

def contrastive_loss(y_true, Dw):
    margin = 1
    a = T.multiply( y_true , T.square(T.maximum(0.0,margin - Dw) ))
    b = T.multiply( T.subtract ( T.constant([1.0],) , y_true ),T.square(Dw))

    '''
    a = T.multiply( y_true , T.pow(T.maximum(0.0,margin - Dw), 2 ))
    b = T.multiply( T.subtract ( T.constant([1.0],) , y_true ), T.pow(Dw,2) )
    '''
    return T.divide(T.add(a,b) , 2 )


def create_pairs(dataset,writers,pair_index):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs =  []
    labels = []
    for writer in range(len(writers)):
        for i in range(len(dataset[writer])-1):
            inc = random.randrange(1, 1556)
            other_writer =  (writer + inc )%1556
            this_writer_image_index=random.randrange(0,len(dataset[writer]))
            if writer!=other_writer:
                other_image_index=random.randrange(0,len(dataset[other_writer]))
                pos_feature1, pos_feature2 = dataset[writer][i], dataset[writer][this_writer_image_index]
                neg_feature1, neg_feature2 = dataset[writer][i], dataset[other_writer][other_image_index]
                pairs += [[pos_feature1, pos_feature2]]
                pairs += [[neg_feature1, neg_feature2]]
                pair_index += [[writer,writer]]
                pair_index += [[writer,other_writer]]
                labels += [0, 1]
            inc = random.randrange(1, 1556)
            other_writer =  (writer + inc )%1556
            this_writer_image_index=random.randrange(0,len(dataset[writer]))
            if writer!=other_writer:
                other_image_index=random.randrange(0,len(dataset[other_writer]))
                pos_feature1, pos_feature2 = dataset[writer][i], dataset[writer][this_writer_image_index]
                neg_feature1, neg_feature2 = dataset[writer][i], dataset[other_writer][other_image_index]
                pairs += [[pos_feature1, pos_feature2]]
                pairs += [[neg_feature1, neg_feature2]]
                pair_index += [[writer,writer]]
                pair_index += [[writer,other_writer]]
                labels += [0, 1]
    print("pairs shape=",np.array(pairs).shape,np.array(labels).shape)
    return np.array(pairs),np.array(labels)

def NeuralNet(pairs,num_epochs):

    pairs = pairs.reshape((-1,2,70,160,1))
    print(pairs.shape)
    # Model
    input_shape = (70,160,1,)
    left_input  = Input(shape = input_shape)

    #left network
    cnn_left=Sequential()
    cnn_left.add(Conv2D(32,(3,3),strides = (2,2),input_shape=input_shape, kernel_initializer='uniform'))
    cnn_left.add(MaxPooling2D(pool_size=(2,2)))
    cnn_left.add(Activation('relu'))


    cnn_left.add(Conv2D(64,(3,3),strides = (2,2) ,kernel_initializer='uniform'))
    cnn_left.add(MaxPooling2D(pool_size=(2,2)))
    cnn_left.add(Dropout(0.2))
    cnn_left.add(Activation('relu'))
    cnn_left.add(Flatten())
    #cnn.add(BatchNormalization())
    cnn_left.add(Dense(32,activation='sigmoid',kernel_initializer='uniform'))

    #right network
    input_shape = (70,160,1,)
    right_input = Input(shape = input_shape)

    cnn_right=Sequential()
    cnn_right.add(Conv2D(32,(3,3),strides = (2,2),input_shape=input_shape, kernel_initializer='uniform'))
    cnn_right.add(MaxPooling2D(pool_size=(2,2)))
    #cnn.add(BatchNormalization())
    cnn_right.add(Activation('relu'))

    cnn_right.add(Conv2D(64,(3,3),strides = (2,2) ,kernel_initializer='uniform'))
    cnn_right.add(MaxPooling2D(pool_size=(2,2)))
    cnn_right.add(Dropout(0.2))
    #cnn.add(BatchNormalization())
    cnn_right.add(Activation('relu'))
    cnn_right.add(Flatten())
    #cnn.add(BatchNormalization())
    cnn_right.add(Dense(32,activation='sigmoid',kernel_initializer='uniform'))
    #

    left_feature=cnn_left(left_input)
    right_feature=cnn_right(right_input)

    L2_layer = Lambda(siamese_euclidean_distance)
    L2_distance = L2_layer([left_feature,right_feature])

#    prediction = Dense(1,activation='sigmoid')(L2_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=L2_distance)

    # train
    adam = Adam(lr=0.001)
    siamese_net.compile(loss=contrastive_loss , optimizer=adam, metrics = ['accuracy'])
    with open('model_architecture.json', 'w') as f:
        f.write(siamese_net.to_json())
    plot_model(siamese_net,to_file="siamese.png",show_shapes=True)
    siamese_net.summary()
    summary = siamese_net.fit([pairs[6000:,0,:,:,:] , pairs[6000:,1,:,:,:] ], labels[6000:], batch_size=128, nb_epoch=num_epochs, validation_data=([pairs[:6000,0,:,:,:], pairs[:6000,1,:,:,:]], labels[:6000]))
    siamese_net.save_weights('model_weights.h5')
    #siamese_net.save('siamese_cnn.h5')
    history_df = pd.DataFrame(summary.history)
    history_df.to_csv("history.csv")
    return siamese_net, history_df
def test(model):
    testimages_path  =    'Hidden Test Folder/DLHiddenTest/DLTestData/'
    pairs_list       =    np.genfromtxt('Hidden Test Folder/DLHiddenTest/DLTestPairs.csv',delimiter=',')
    pairs_df         =    pd.read_csv('Hidden Test Folder/DLHiddenTest/DLTestPairs.csv')
    test_pairs = []
    for i in range(pairs_df.shape[0]):
        # Read and normalize images
        img1 = norm_resize(cv.imread(testimages_path + str(pairs_df['FirstImage'][i]),0))
        img2 = norm_resize(cv.imread(testimages_path + str(pairs_df['SecondImage'][i]),0))
        #Form pairs
        test_pairs +=[[img1,img2]]
    test_pairs = np.array(test_pairs).reshape((-1,2,70,160,1))
    pred = model.predict([test_pairs[:,0,:,:,:],test_pairs[:,1,:,:,:]],batch_size=128)
    rounded = [np.round(x) for x in pred]
    S_D = []
    for x in rounded:
        if x == 1:
            S_D += ['Different']
        else:
            S_D += ['Same']
    pairs_df['SameOrDifferent'] = S_D
    print(pairs_df)
    pairs_df.to_csv('Hidden Test Folder/DLHiddenTest/DLTestOutput.csv',sep=',')

dataset,writers =   dataset_init()
pair_index  =   []
pairs,labels=create_pairs(dataset,writers,pair_index)
model,history =  NeuralNet(pairs,num_epochs = 30)
test(model)
