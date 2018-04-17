import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt
import  warnings
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

num_trees=100

test_size=0.10

seed=9



fixed_size=(250,250)

fearow=[0 for x in range(100)]
labrow=[0 for x in range(100)]
meanrow=[0 for x in range(100)]

n=0
rowfeature=[]
with open('output 4classes/data_csv.csv','r') as File:  
    reader = csv.reader(File)
    for rows in reader:
        fearow[n]=np.array(rows)
        n=n+1
        rowfeature.append(rows)
#x=np.array(row)
#print(x)
print('fearow[0]')
#print(fearow[0])


rowlabel=[]
n=0
with open('output 4classes/labels_csv.csv','r') as File:  
    reader = csv.reader(File)
    for rows in reader:
        labrow[n]=np.array(rows)
        n=n+1
        rowlabel.append(rows)


values_of_features=fearow[0]
i=1
while i<24:
    values_of_features=np.append(values_of_features,fearow[i])
    i=i+1

values_of_labels=labrow[0]
i=1
while i<24:
    values_of_labels=np.append(values_of_labels,labrow[i])
    i=i+1




print('3:top floor  2:mtech floor 1:hod  0:ground')
#print(np.array(fearow))

print('.............SUPPORT VECTOR MACHINES.............')
from sklearn import svm


lin_clf = svm.SVC()
lin_clf.fit(np.array(fearow), np.array(labrow).ravel())


clf  = RandomForestClassifier()
clf.fit(np.array(fearow), np.array(labrow).ravel())



linear=linear_model.LogisticRegression()
linear.fit(np.array(fearow), np.array(labrow).ravel())

knn = KNeighborsClassifier()
knn.fit(np.array(fearow), np.array(labrow).ravel())



def featured_global(image, mask=None):
    #print('entered global')
    fg= np.fft.fft2(image)
    fg=np.abs(fg)
    X=np.array(fg)
    mean_vec=np.mean(X,axis=0)
    cov_mat = np.cov(X.T)
    cov_mat=(X-mean_vec).T.dot((X-mean_vec))/(X.shape[0]-1)
    eig_vals1, eig_vecs1 = np.linalg.eig(cov_mat)
    eig_vals1=np.abs(eig_vals1)
    #print('end global')
    return eig_vals1

def featured_local(image, mask=None):
    #print('entered local')
    a,b,m=0,0,0
    p,q=0,0
    fg= [[0 for x in range(10)] for y in range(10)]
    M_selection= [[0 for x in range(10)] for y in range(10)]
    eig_vals=[0 for x in range(500)]
    eig_vecs=[0 for x in range(500)]
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
    #print('end local')
    return pcavalues




testfearow=[0 for x in range(1)]

test_path = "dataset/testing folder images"

train_path="dataset/training folder images"

train_labels=os.listdir(test_path)

for file in glob.glob(test_path + "/*.jpg"):
    image=cv2.imread(file)
    print('file',file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('original image',image)
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0   #Identity, times two! 
    boxFilter = np.ones( (9,9), np.float32) / 81.0
    kernel = kernel - boxFilter
    image = cv2.filter2D(image, -1, kernel)
    image=cv2.resize(image, fixed_size)
    print('entered')
    #cv2.imshow('image',image)
    gl_fea=featured_global(image)
    lo_fea=featured_local(image)

    all_features=np.hstack([gl_fea,lo_fea])
    predicted_label = lin_clf.predict (all_features.reshape(1,-1))[0]
    print('support vector machine label is',predicted_label)
    
    predicted_label =clf.predict(all_features.reshape(1,-1))[0]
    print('random forest classifier label is',predicted_label)
    
    predicted_label= linear.predict(all_features.reshape(1,-1))[0]
    print('Logistic Regression label is:',int(predicted_label))

    predicated_label = knn.predict(all_features.reshape(1,-1))[0]
    print('knn label is:',int(predicated_label))
    if int(predicted_label)==0:
        print('image is taken from 50feet')
    elif int(predicted_label)==1:
        print('image is taken from 40 feet')
    elif int(predicted_label)==2:
        print('image is taken from 30 feet')
    elif int(predicted_label)==3:
        print('image is taken from 15 feet')
