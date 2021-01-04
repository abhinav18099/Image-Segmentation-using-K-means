#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 09:40:34 2020

@author: abhinav
"""
##  importting libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import pathlib
###################################################
# created pixel.csv to work on the image and feature
path = str(pathlib.Path(__file__).parent.absolute())
'''
img = Image.open(path+'/Image.jpg', 'r')
x=[]
y=[]
r=[]
g=[]
b=[]
for i in range(img.width):
    for j in range(img.height):
        x.append(i)
        y.append(j)
        k=img.getpixel((i,j))
        r.append(k[0])
        g.append(k[1])
        b.append(k[2])
df=pd.DataFrame()
df['x']=x
df['y']=y
df['r']=r
df['g']=g
df['b']=b
df.to_csv(path+'/pixel.csv',index=False)
'''
####################################################
# loading the csv file into dataframe

df = pd.read_csv(path+"/pixel.csv")
print(df.head(10))

df1 = df[['r','g','b','y','x']]
img = cv2.imread(path+'/Image.jpg')
img = img.astype('uint8')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
######################################################
######################################################
image = np.float32(img.reshape((-1,3)))


##############################################

### cv2 library Kmeans used in below code #####

## adding width location of a pixel at index 3
## adding height location of pixel at index 4

img_dist = np.zeros([int(image.size/3),5])
for i in range (0, int(image.size/3)):
    img_dist[i][0] = image[i][0]
    img_dist[i][1] = image[i][1]
    img_dist[i][2] = image[i][2]
img_dist = np.float32(img_dist)

## min -max normalisation of i is taken care of
## during assignment in below code

for i in range (0,int(len(img))):
    for j in range(0,int((img.size/(3*len(img))))):
        img_dist[int((img.size/(3*len(img))))*i + j][3] = i*(255)/(int(len(img))) 
        img_dist[int((img.size/(3*len(img))))*i + j][4] = j*(255)/(int((img.size/(3*len(img)))))
        
## Applying K-means algorithm using library

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1.0)
### number of cluster is taken as 10

k = 10
ret, label, center =  cv2.kmeans(img_dist, k, None, criteria, 10,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
#Reshaping the image
final_image = np.uint8(np.zeros([int(image.size/3),3]))

for i in range (0, int(image.size/3)):
    final_image[i][0] = res[i][0]
    final_image[i][1] = res[i][1]
    final_image[i][2] = res[i][2]
result = np.uint8(final_image.reshape((img.shape)))
plt.imshow(result)
plt.show()

##############################################################
## Applying K-means algorithm without cv2 Kmeans 


K_clf1 = KMeans(2,random_state=42)
K_clf1.fit(df.values)
label_val1 = K_clf1.labels_
center1 = K_clf1.cluster_centers_
center1 = np.delete(center1,[0,1],1)
center1 = center1.astype(int);

res1 = center1[label_val1]
result_image1 = res1.reshape((img.shape))
plt.imshow(result_image1)
plt.show()

#################################################################
