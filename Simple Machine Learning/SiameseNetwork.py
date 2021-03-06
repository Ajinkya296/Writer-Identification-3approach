import numpy as np
from featureExtractor import Extractor
np.random.seed(1337) # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input
from keras.optimizers import SGD, RMSprop ,Adam
from keras import backend as K
from keras.callbacks import History
from keras.utils import plot_model
import tensorflow as T
import os,random
import collections
import matplotlib.pyplot as plt
import pandas as pd
n = 1556
feature_size = 32

#----------------------------------------------------------------------------------------------------------------------------------
'''
Calculates contrastive Loss
'''
def contrastive_loss(y_true, Dw):
    margin = 1
    a = T.multiply( y_true , T.square(T.maximum(0.0,margin - Dw) ))
    b = T.multiply( T.subtract ( T.constant([1.0],) , y_true ),T.square(Dw))
    return T.divide(T.add(a,b) , 2 )

#----------------------------------------------------------------------------------------------------------------------------------
'''
Splits the input in two halves
Computes Euclidean distance between the two halves
'''
def siamese_euclidean_distance(pair):

    feature1,feature2 = pair[:,:feature_size],pair[:,feature_size:]
    print("Euclidean Input Shape : " , feature1.get_shape().as_list())
    diff = T.subtract(feature1,feature2)
    sqr  = T.pow(diff,2)
    print("Squared Shape : " , sqr.get_shape().as_list())
    sum_ = T.reduce_sum(sqr,1,keep_dims=True)
    print("Mean Shape : ",sum_.get_shape().as_list())
    dist = T.sqrt(sum_)
    print("Dist Shape : ",dist.get_shape().as_list())
    return dist

#----------------------------------------------------------------------------------------------------------------------------------
def normalized(v):
    mx = np.max(v)
    mn = np.min(v)
    return np.divide(v,float(mx-mn))
#----------------------------------------------------------------------------------------------------------------------------------
'''
Reads the csv file of features generated by featuresExtractor.py
Reads the image names to extract writer names
Normalizes the features and forms dataset(i.e list of normalized features for every writer provided)
'''
def read_features(path):
    features = np.genfromtxt('features.csv',delimiter = ',')
    ext      = Extractor()
    features     = ext.extract_folder('Images')
    train_path = path
    training_names = os.listdir(train_path)
    dataset = [[] for i in  range(n)]
    writers = []
    curr_index = -1
    i = 0
    for name in training_names:
        writer_name = name[:4]
        if writer_name not in writers:
            writers.append(writer_name)
            curr_index += 1
        dataset[curr_index].append(normalized(features[i]))
        i += 1
    return np.array(dataset),writers

#----------------------------------------------------------------------------------------------------------------------------------
'''
Positive and negative pair creation.
Alternates between positive and negative pairs.
For every writer choose a random image of the same writer.
For every random other writer choose a random image of the writer.
Repeat this process to increase the size of the dataset. [Note: Since choice of images is random ,looping does not necessarily add duplication]
'''
def create_pairs(data,writers,pair_index):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''


    pairs =  []
    labels = []
    for writer in range(len(writers)):
        for i in range(len(data[writer])-1):
            for k in range(7):
                inc = random.randrange(1, 1556)
                other_writer =  (writer + inc )%1556
                if writer!=other_writer:
                    t = random.randrange(0,len(data[writer])-1)
                    pos_feature1, pos_feature2 = data[writer][i], data[writer][t]
                    neg_feature1, neg_feature2 = data[writer][i], data[other_writer][ random.randrange(0,len(data[other_writer]))]

                    pairs += [[pos_feature1, pos_feature2]]
                    pairs += [[neg_feature1, neg_feature2]]

                    pair_index += [[writer,writer]]
                    pair_index += [[writer,other_writer]]
                    labels += [0, 1]

    summary = pd.DataFrame( np.column_stack((np.array(pair_index),np.array(labels))) , columns = ['writer1' ,'writer2','label'])
    return np.array(pairs), np.array(labels)
#----------------------------------------------------------------------------------------------------------------------------------
'''
Simple Neural Network model
'''
def NeuralNet(pairs,num_epochs):
    # Model
    input_shape = (2*feature_size,)
    input       = Input(shape = input_shape)

    L2_layer = Lambda(siamese_euclidean_distance)
    L2_distance = L2_layer(input)
    prediction = Dense(1,activation='sigmoid')(L2_distance)

    siamese_net = Model(inputs=input,outputs=prediction)


    # Train
    siamese_net.compile(loss=contrastive_loss , optimizer='rmsprop', metrics = ['accuracy'])
    history = History()


    train_input = []
    test_input = []
    # Concatenate both features in one feature vector which will be split in half while calculating distance between them
    test_frac = 0.2
    test_size = round(test_frac*pairs.shape[0])
    for i in range(test_size,pairs.shape[0]):
        train_input += [np.concatenate((pairs[i,0], pairs[i,1])).tolist()]
    for j in range(0,test_size):
        test_input += [np.concatenate((pairs[j,0], pairs[j,1])).tolist()]

    train_input = np.array(train_input)
    test_input  = np.array(test_input)

    summary = siamese_net.fit(train_input, labels[test_size:], batch_size=128, nb_epoch = num_epochs, callbacks= [history],validation_data = (test_input,labels[:test_size]))
    siamese_net.summary()
    history_df = pd.DataFrame(summary.history)
    history_df.to_csv("history.csv")
    return siamese_net, history_df

#----------------------------------------------------------------------------------------------------------------------------------

def test(model):
    test_pairs = []
    testimages_path  =    'Hidden Test Folder/MLHiddenTest/MLTestData/'
    ex = Extractor()
    test_features = ex.extract_test(testimages_path)
    i = 0
    feat_dict = {}
    for name in os.listdir(testimages_path):
        feat_dict[name] = test_features[i]
        i+=1
    pairs_list =    np.genfromtxt('Hidden Test Folder/MLHiddenTest/MLTestPairs.csv',delimiter=',')
    pairs_df   =    pd.read_csv('Hidden Test Folder/MLHiddenTest/MLTestPairs.csv')
    for i in range(pairs_df.shape[0]):
        img1 = feat_dict[pairs_df['FirstImage'][i]]
        img2 = feat_dict[pairs_df['SecondImage'][i]]
        test_pairs +=[[img1,img2]]
    test_pairs = np.array(test_pairs)
    test_input = []
    for i in range(test_pairs.shape[0]):
        test_input += [np.concatenate((test_pairs[i,0], test_pairs[i,1])).tolist()]
    pred = model.predict(np.array(test_input),batch_size=128).reshape((test_pairs.shape[0],))
    print(pred.shape)
    print(pred)
    rounded = [np.round(x) for x in pred]
    S_D = []
    for x in rounded:
        if x == 1:
            S_D += ['Different']
        else:
            S_D += ['Same']
    pairs_df['SameOrDifferent'] = S_D
    print(pairs_df)
    pairs_df.to_csv('Hidden Test Folder/MLHiddenTest/MLTestOutputs.csv',sep=',')

data,writers =  read_features('Images/')
pair_index = []
pairs , labels =  create_pairs(data,writers,pair_index)

model,history =  NeuralNet(pairs,num_epochs = 10)
test(model)
plot_model(model, to_file='model.png' ,show_shapes = True)

# Plot the graphs
plt.figure(1)
axes = plt.gca()
axes.set_ylim([0,1])
history['acc'].plot()

plt.figure(2)
history['loss'].plot()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------
