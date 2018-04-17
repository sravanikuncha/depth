import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
from PIL import Image
from os import listdir
from os.path import isfile, join
import glob
import numpy
import cv2
import h5py


fixed_size=tuple((250,250))

test_size=0.10
train_path="dataset/training folder images"
train_labels=os.listdir(train_path)
print(train_labels)#o/p:['0'.....'3']

features=[]
features1=[]
features2=[]
labels=[]



i,j=0,0
e=0
seed=9
images_per_class=25

def featured_global(image, mask=None):
    print('entered global')
    fg= np.fft.fft2(image)
    fg=np.abs(fg)
    X=np.array(fg)
    mean_vec=np.mean(X,axis=0)
    cov_mat = np.cov(X.T)
    cov_mat=(X-mean_vec).T.dot((X-mean_vec))/(X.shape[0]-1)
    eig_vals1, eig_vecs1 = np.linalg.eig(cov_mat)
    eig_vals1=np.abs(eig_vals1)
    print('end global')
    return eig_vals1

def featured_local(image, mask=None):
    print('entered local')
    a,b,m=0,0,0
    p,q=0,0
    fg= [[0 for x in range(10)] for y in range(10)]
    M_selection= [[0 for x in range(10)] for y in range(10)]
    eig_vals=[0 for x in range(100)]
    eig_vecs=[0 for x in range(100)]
    while p<250 or q<250:
       
        i=p
        j=i+25
        k=q
        l=k+25
        M_selection[a][b] = image[i:j, k:l]
        fg[a][b]= np.fft.fft2(M_selection[a][b])
        fg[a][b]=np.abs(fg[a][b])
        X=np.array(fg[a][b])
        mean_vec=np.mean(X,axis=0)
        cov_mat = np.cov(X.T)
        cov_mat=(X-mean_vec).T.dot((X-mean_vec))/(X.shape[0]-1)
        eig_vals[m], eig_vecs[m] = np.linalg.eig(cov_mat)
        eig_vals[m]=np.abs(eig_vals[m])
        m=m+1
        q=q+25
        b=b+1
        if l==250 and p!=250:
            if l==250 and p!=225:
                q=0
                b=0
            p=p+25
            a=a+1

    pcavalues=eig_vals[0]
    o=1
    while o<100:
        pcavalues=np.append(pcavalues,eig_vals[o])
        o=o+1
    print('end local')
    return pcavalues


yy=0

fea=[0 for x in range(100)]
for training_name in train_labels:
    dir=os.path.join(train_path,training_name)
     
    current_label=training_name

    e=1
    for xx in range(1,images_per_class+1):
        file=dir+"/"+str(xx)+".jpg"
        print('file',file)
        image=cv2.imread(file)
        image=cv2.resize(image, fixed_size)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kernel = np.zeros( (9,9), np.float32)
        kernel[4,4] = 2.0   #Identity, times two! 
        boxFilter = np.ones( (9,9), np.float32) / 81.0
        kernel = kernel - boxFilter
        image = cv2.filter2D(image, -1, kernel)


        gl_fea=featured_global(image)
        lo_fea=featured_local(image)

        all_features=np.hstack([gl_fea,lo_fea])
        #print(all_features)

        labels.append(current_label)
        fea[yy]=all_features
        yy=yy+1
        features.append(all_features)
        if yy>25:
           features1.append(all_features)
        if yy>50:
            features2.append(all_features)
        i+=1
        e+=1
    print("[status] processed folder:{}",format(current_label))
    j+=1
print("[status] completed all features extraction")

#print(features)


csvfile = "output 4classes/data_csv.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output1:
    writer = csv.writer(output1, lineterminator='\n')
    writer.writerows(features)

csvfile="output 4classes/labels_csv.csv"


with open(csvfile, "w") as output1:
    writer = csv.writer(output1, lineterminator='\n')
    writer.writerows(labels)







csvfile = "output 4classes/data_csv_0andall.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output1:
    writer = csv.writer(output1, lineterminator='\n')
    writer.writerows(features)

print('done 0and all')

csvfile = "output 4classes/data_csv_1andall.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output1:
    writer = csv.writer(output1, lineterminator='\n')
    writer.writerows(features1)

print('done : 1 andall')


csvfile = "output 4classes/data_csv_2andall.csv"


#Assuming res is a list of lists
with open(csvfile, "w") as output1:
    writer = csv.writer(output1, lineterminator='\n')
    writer.writerows(features2)


print(' done:2 and all')

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(features)

h5f_data = h5py.File('output 4classes/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output 4classes/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

