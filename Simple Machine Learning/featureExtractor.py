import numpy as np
import cv2
from matplotlib import pyplot as plt
import os,random
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

class Extractor:


    def extract_test(self,path):
        sift = cv2.xfeatures2d.SIFT_create()
        train_path = path +'/'
        training_names = os.listdir(train_path)
        image_paths = []
        image_classes = []
        class_id = 0
        k = 32
        for training_name in training_names:
            dir = os.path.join(train_path, training_name)
            image_paths.append(dir)

        des_list = []
        for image_path in image_paths:
            im = cv2.imread(image_path,0)
            kpts, des = sift.detectAndCompute(im,None)
            des_list.append((image_path, des))

        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        im_features = np.zeros((len(image_paths), k), "float32")

        voc = np.load('centers.npy')
        for i in range(len(training_names)):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                #features[ training_names[i] ][ w ] += 1
                im_features[i][w] += 1

        feat = np.array(im_features)
        print('Features Extracted for testing')
        #np.savetxt(export_name,im_features, delimiter=',')
        return feat

    def extract_folder(self,path):
        sift = cv2.xfeatures2d.SIFT_create()
        train_path = path +'/'
        training_names = os.listdir(train_path)
        image_paths = []
        image_classes = []
        class_id = 0
        for training_name in training_names:
            dir = os.path.join(train_path, training_name)
            image_paths.append(dir)

        des_list = []
        for image_path in image_paths:
            im = cv2.imread(image_path,0)
            kpts, des = sift.detectAndCompute(im,None)
            des_list.append((image_path, des))

        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))

        print(len(descriptors))
        k = 32
        print("Starting Kmeans")
        voc, variance = kmeans(descriptors, k, 1)
        print("Kmeans done")
        np.save('centers.npy',voc)
        im_features = np.zeros((len(image_paths), k), "float32")
        for i in range(len(training_names)):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                #features[ training_names[i] ][ w ] += 1
                im_features[i][w] += 1

        feat = np.array(im_features)
        print('Features Extracted')
        #np.savetxt(export_name,im_features, delimiter=',')
        return feat
