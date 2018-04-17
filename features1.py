import h5py
import numpy as np
import os
import glob
import csv
import cv2
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from os import listdir
from sklearn.externals import joblib

num_trees=100

test_size=0.10

seed=9

fearow=[0 for x in range(100)]
labrow=[0 for x in range(100)]
fixed_size=(250,250)
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
# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output 4classes/data.h5', 'r')
h5f_label = h5py.File('output 4classes/labels.h5', 'r')
#h5f_mean = h5py.File('output/means.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']
#global_means_string = h5f_mean['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
#global_means = np.array(global_means_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")


print('value of features',global_features)

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=test_size,random_state=seed)

                                                                                          
                                                                                

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


import warnings
warnings.filterwarnings('ignore')


for name, model in models:
    kfold = KFold(n_splits=6, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()





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
    print('end local')
    return pcavalues



print('.......RANDOM FOREST CLASSIFIER.............')


# path to test data
test_path = "dataset/testing folder images"

train_path="dataset/training folder images"

train_labels=os.listdir(test_path)

for file in glob.glob(test_path + "/*.jpg"):
    image=cv2.imread(file)
    image=cv2.resize(image, fixed_size)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('original image',image)
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0   #Identity, times two! 
    boxFilter = np.ones( (9,9), np.float32) / 81.0
    kernel = kernel - boxFilter
    image = cv2.filter2D(image, -1, kernel)
    print('entered')
    #cv2.imshow('image',image)
    gl_fea=featured_global(image)
    lo_fea=featured_local(image)

    all_features=np.hstack([gl_fea,lo_fea])
    print('Random forest classifier')
    clf  = RandomForestClassifier()
    clf.fit(np.array(fearow), np.array(labrow).ravel())
    prediction = clf.predict(all_features.reshape(1,-1))[0]
    print('image can be from floor no:',prediction)
    
    

