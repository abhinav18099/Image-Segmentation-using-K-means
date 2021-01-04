import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pathlib

## setting path variable to run on any machine
path = str(pathlib.Path(__file__).parent.absolute())
path1 = path+"/non_linearly_seperable data/Class1.txt"
path2 = path+"/non_linearly_seperable data/Class2.txt"

## reading csv of class1 in dataframe df

df = pd.read_csv(path1,sep="\t",header= None,index_col=False)
df.columns = ['A', 'B', 'C']
df.drop('C',axis=1,inplace=True)

## reading csv of class1 in dataframe df1

df1 = pd.read_csv(path2,sep="\t",header= None,index_col=False)
df1.columns = ['A', 'B', 'C']
df1.drop('C',axis=1,inplace=True)

# splitting the data into train and test set

train_class1,test_class1,train_class2,test_class2 = train_test_split(df,df1,test_size=0.3,random_state=42)

## plotting the data
x=[]
y=[]
for i in train_class1.values:
   x.append(i[0])
   y.append(i[1])

x2=[]
y2=[]
for i in train_class2.values:
   x2.append(i[0])
   y2.append(i[1])
   
plt.scatter(x,y,label='class1')
plt.scatter(x2,y2,label='class2')
plt.legend()
plt.show()

## joining the training data for class1 and class 2 for feeding into Kmeans
#print(train_class1)
#print(train_class2)
X_train = pd.concat([train_class1,train_class2])
X_test = pd.concat([test_class1,test_class2])
X_train_label = []
X_test_labels = []
for i in range(len(train_class1)):
        X_train_label.append(1)
for i in range(len(train_class2)):
        X_train_label.append(0)
#print(X_train)
for i in range(len(test_class1)):
        X_test_labels.append(1) 
for i in range(len(test_class1)):
        X_test_labels.append(0) 
        
## initialising mean for class1 and class2

x1=0
y1=0
x0=1
y0=1
n=len(X_train)
x_train_cluster=[0 for i in range(n)]
ds=0
co=0
for i,j in zip(X_train['A'],X_train['B']) :
    dis0=(i-x0)**2+(j-y0)**2
    dis1=(i-x1)**2+(j-y1)**2
    if(dis0<dis1):
        ds+=dis0
        x_train_cluster[co]=0
    else:
        ds+=dis1
        x_train_cluster[co]=1
    co+=1
ds1=ds
## distortion measure when initial mean value is taken 

## calulating new mean and
## looping until there is an convergence
while(True):
    n0=0
    n1=0
    x1=0
    y1=0
    x0=0
    y0=0
    co=0
    for i,j in zip(X_train['A'],X_train['B']) :
        if(x_train_cluster[co]==0):
            x0+=i
            y0+=j
            n0+=1
        else:
            x1+=i
            y1+=j
            n1+=1
        co+=1
    x1/=n1
    y1/=n1
    x0/=n0
    y0/=n0
    co=0
    ds=0
    for i,j in zip(X_train['A'],X_train['B']) :
        dis0=(i-x0)**2+(j-y0)**2
        dis1=(i-x1)**2+(j-y1)**2
        if(dis0<dis1):
            ds+=dis0
            x_train_cluster[co]=0
        else:
            ds+=dis1
            x_train_cluster[co]=1
        co+=1
    if((ds1-ds)<0.5):
        break
    ds1=ds
x_test_cluster=[0 for i in range(len(X_test))]
co=0
for i,j in zip(X_test['A'],X_test['B']) :
    dis0=(i-x0)**2+(j-y0)**2
    dis1=(i-x1)**2+(j-y1)**2
    if(dis0<dis1):
            
        x_test_cluster[co]=0
    else:
            
        x_test_cluster[co]=1
    co+=1
n1=len(X_train_label)
## varibles for getting values in confusion matrix

co=0
tp=0
fp=0
tn=0
fn=0
for i in range(n1):
    if(x_train_cluster[i]==X_train_label[i]):
        co+=1
    if(x_train_cluster[i]==0 and X_train_label[i]==0):
        tp+=1
    if(x_train_cluster[i]==0 and X_train_label[i]==1):
        fp+=1
    if(x_train_cluster[i]==1 and X_train_label[i]==0):
        fn+=1
    if(x_train_cluster[i]==1 and X_train_label[i]==1):
        tn+=1
print("accuracy score for train set :",co/n1)
print("confusion matrix for the following classfication ")
co_matrix=np.matrix([[tp,fp],[fn,tn]])
print(co_matrix)
plt.scatter(X_train['A'],X_train['B'],c=x_train_cluster)
plt.title("training set")
plt.show()
n1=len(X_test)

## confusion matrix for class2
co=0
tp=0
fp=0
tn=0
fn=0
for i in range(n1):
    if(x_test_cluster[i]==X_test_labels[i]):
        co+=1
    if(x_test_cluster[i]==0 and X_test_labels[i]==0):
        tp+=1
    if(x_test_cluster[i]==0 and X_test_labels[i]==1):
        fp+=1
    if(x_test_cluster[i]==1 and X_test_labels[i]==0):
        fn+=1
    if(x_test_cluster[i]==1 and X_test_labels[i]==1):
        tn+=1
print("accuracy score for test set :",co/n1)
print("confusion matrix for the following classfication ")
co_matrix=[[tp,fp],[fn,tn]]
print(np.matrix(co_matrix))
plt.scatter(X_test['A'],X_test['B'],c=x_test_cluster)
plt.title("test set")
plt.show()