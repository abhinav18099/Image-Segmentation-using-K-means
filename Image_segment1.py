#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 09:40:34 2020

@author: abhinav
"""
##  importting libraries
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import pandas as pd
import pathlib
###################################################
# created pixel.csv to work on the image and feature
path = str(pathlib.Path(__file__).parent.absolute())
"""
img = Image.open('/home/abhinav/Downloads/CS669/assignment_2/Assignment_2/Image.jpg', 'r')
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
df.to_csv('/home/abhinav/Downloads/CS669/assignment_2/Assignment_2/pixel.csv',index=False)
""" 
####################################################
# loading the csv file into dataframe

df = pd.read_csv(path+"/pixel.csv")
print(df.head(10))
## real image plot using cv2
### cv2 is used only to read the pixel value not to perform Kmeans
img = cv2.imread(path+"/Image.jpg")
img = img.astype('uint8')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

######################################################

## applying K-means algorithm only on pixel value 
### two cluster is taken 
K_clf = KMeans(2,random_state=42,max_iter=10)
size = img.shape
img = img.reshape((img.shape[0] * img.shape[1], 3))
K_clf.fit(img)
label_val = K_clf.labels_
center = K_clf.cluster_centers_
center = center.astype(int);

res = center[label_val]
result_image = res.reshape((size))

plt.imshow(result_image)
plt.show()
'''
plt.scatter(df['x'],df['y'],c=label_val)
plt.show()
'''
####################################################

















