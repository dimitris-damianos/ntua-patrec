from utils import data_parser,extract_mfcc, extract_mfsc,digitsHistogram,legend_without_duplicate_labels,createMfccDataset,createFeatsDataset,split_data
from models import myRNN,myLSTM,myGRU
from training import training

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from sklearn.decomposition import PCA
import sklearn.naive_bayes as nb
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dec_tree
from sklearn.metrics import accuracy_score
import os

import torch

### STEP 2 ####
print("\n### STEP 2 ###\n")
loc = '../pr_lab1_2020-21_data/digits'
waves, ids, fnames = data_parser(loc)


### STEP 3 ###
print("\n### STEP 3 ###\n")
mfccs, deltas, ddeltas = extract_mfcc(waves)
mfscs = extract_mfsc(waves)

### STEP 4 ###
print("\n### STEP 4 ###\n")
n1 = 5 # Damianos
n2 = 4 # Aravanis

ston = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

waves, ids, digits = data_parser(loc)
mfccs, deltas, ddeltas = extract_mfcc(waves)

digits = [ston[d] for d in digits]

mfscs = extract_mfsc(waves)

## plt histograms of selected digits
digitsHistogram(n1,n2,mfccs,digits)

# Find two transcriptions for n1

i = 0
two_n1 = []
for idx, value in enumerate(digits):
    if value == n1:
        two_n1.append(idx)
        i +=1
        if i == 2:
            break

corr_matrix = np.corrcoef(np.concatenate((mfccs[two_n1[0]],mfccs[two_n1[1]]), axis=0).T)
fig, ax = plt.subplots(figsize=(6,6))
sn.heatmap(corr_matrix, ax=ax)
plt.title(f"Correlation of MFCCs for {n1}")
plt.show()

corr_matrix = np.corrcoef(np.concatenate((mfscs[two_n1[0]],mfscs[two_n1[1]]), axis=0).T)
fig, ax = plt.subplots(figsize=(6,6))
sn.heatmap(corr_matrix, ax=ax)
plt.title(f"Correlation of MFSCs for {n1}")
plt.show()
        
# Find two transcriptions for n2
i = 0
two_n2 = []
for idx, value in enumerate(digits):
    if value == n2:
        two_n2.append(idx)
        i +=1
        if i == 2:
            break

corr_matrix = np.corrcoef(np.concatenate((mfccs[two_n2[0]],mfccs[two_n2[1]]), axis=0).T)
fig, ax = plt.subplots(figsize=(6,6))
sn.heatmap(corr_matrix, ax=ax)
plt.title(f"Correlation of MFCCs for {n2}")
plt.show()  

corr_matrix = np.corrcoef(np.concatenate((mfscs[two_n2[0]],mfscs[two_n2[1]]), axis=0).T)
fig, ax = plt.subplots(figsize=(6,6))
sn.heatmap(corr_matrix, ax=ax)
plt.title(f"Correlation of MFSCs for {n2}")
plt.show()

### STEP 5 ###
print("\n### STEP 5 ###\n")
features = [np.concatenate((mfccs[i],deltas[i],ddeltas[i]), axis=1) for i in range(len(mfccs))]

feature_vectors = [np.concatenate((np.mean(features[i], axis=0),np.std(features[i], axis=0))) 
                  for i in range(len(features))]

ntoc = {1:'g', 2:'r', 3:'c', 4:'maroon', 5:'y',
        6:'k', 7:'m', 8:'aqua', 9:'b'}

for idx, vec in enumerate(feature_vectors):
    x = vec[0]
    y = vec[1]
    plt.scatter(x,y,color=ntoc[digits[idx]],label=digits[idx])
    
legend_without_duplicate_labels(plt)
plt.title('Scatter before PCA')
plt.show()

### STEP 6 ### 
print("\n### STEP 6 ###\n")
data = np.asarray(feature_vectors)
#print(data.shape)

pca_2 = PCA(n_components=2)
pca_3 = PCA(n_components=3)

# Two dim PCA

two_dim = pca_2.fit_transform(data)

print(f"The variance of the two components : {pca_2.explained_variance_ratio_}")

for idx, vec in enumerate(two_dim):
    x = vec[0]
    y = vec[1]
    plt.scatter(x,y,color=ntoc[digits[idx]],label=digits[idx])
    
legend_without_duplicate_labels(plt)
plt.title('Scatter after PCA (dim=2)')
plt.show()

# Three dim PCA

three_dim = pca_3.fit_transform(data)
#print(three_dim.shape)

print(f"The variance of the three components : {pca_3.explained_variance_ratio_}")

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
    
for idx, vec in enumerate(three_dim):
    x = vec[0]
    y = vec[1]
    z = vec[2]
 
    ax.scatter3D(x, y, z, color=ntoc[digits[idx]],label=digits[idx])

    
legend_without_duplicate_labels(plt)
plt.title('Scatter after PCA (dim=3)')
plt.show()


### STEP 7 ###
print("\n### STEP 7 ###\n")
## Using MFCC features:
dataset = createMfccDataset(features,digits)
x_train, y_train, x_test, y_test = split_data(dataset)

#### Model training ####
print('Training classifiers using MFCCs and their deltas.')

## 1st model: gaussian naive bayes model training and testing
print('Training Naive Bayes...')
gnb = nb.GaussianNB()
gnb.fit(x_train,y_train)

gnb_predictions = gnb.predict(x_test)

## 2nd mosel: MLP
print("Training MLP...")
mlp_model = mlp(random_state=1, max_iter=500)
mlp_model.fit(x_train, y_train)
mlp_predictions = mlp_model.predict(x_test)

## 3rd model: k-nn
print("Training k-nn...")
knn_model = knn(n_neighbors=13)
knn_model.fit(x_train,y_train)
knn_predictions = knn_model.predict(x_test)

## 4th model: decision tree 
print("Training decision tree...")
decTree_model = dec_tree()
decTree_model.fit(x_train, y_train)
decTree_predictions = decTree_model.predict(x_test)

### Test the quality of predictions

## Check accuracy:
print('Models accuracy:')
gnb_acc = accuracy_score(y_test, gnb_predictions)
mlp_acc = accuracy_score(y_test, mlp_predictions)
knn_acc = accuracy_score(y_test, knn_predictions)
decTree_acc = accuracy_score(y_test, decTree_predictions)

print(f"Naive Bayes: {gnb_acc}\nMLP: {mlp_acc} \nK-NN: {knn_acc} \nDecision Tree: {decTree_acc}")

### Testing using the created feature vectors:
dataset = createFeatsDataset(feature_vectors,digits)
x_train, y_train, x_test, y_test = split_data(dataset)

'''print(f'Percentages: train:{len(x_train)/len(dataset)}, test:{len(x_test)/len(dataset)}')
print(f'Data shape:{np.shape(x_test)}')
print(f'Labels shape:{np.shape(y_test)}')'''

#### Model training ####
print('\nTraining classifiers using mean and variance features.')

## 1st model: gaussian naive bayes model training and testing
print('Training Naive Bayes...')
gnb = nb.GaussianNB()
gnb.fit(x_train,y_train)

gnb_predictions = gnb.predict(x_test)

## 2nd mosel: MLP
print("Training MLP...")
mlp_model = mlp(random_state=1, max_iter=500)
mlp_model.fit(x_train, y_train)
mlp_predictions = mlp_model.predict(x_test)

## 3rd model: k-nn
print("Training k-nn...")
knn_model = knn(n_neighbors=13)
knn_model.fit(x_train,y_train)
knn_predictions = knn_model.predict(x_test)

## 4th model: decision tree 
print("Training decision tree...")
decTree_model = dec_tree()
decTree_model.fit(x_train, y_train)
decTree_predictions = decTree_model.predict(x_test)

### Test the quality of predictions

## Check accuracy:
print('Models accuracy:')
gnb_acc = accuracy_score(y_test, gnb_predictions)
mlp_acc = accuracy_score(y_test, mlp_predictions)
knn_acc = accuracy_score(y_test, knn_predictions)
decTree_acc = accuracy_score(y_test, decTree_predictions)

print(f"Naive Bayes: {gnb_acc}\nMLP: {mlp_acc} \nK-NN: {knn_acc} \nDecision Tree: {decTree_acc}")


### STEP 8 ###
print("\n### STEP 8 ###\n")

## create sin and cos sequences to sample from
freq = 40
points = 1000 ## points per time
max_t = 1
t = np.linspace(0,max_t,max_t*points)
sin_t = np.sin(2*np.pi*freq*t)
cos_t = np.cos(2*np.pi*freq*t)

## divide sin and cos to sub-sequences
frame = 10
start = 0
sin_samp = []
cos_samp = []
while start+frame<len(sin_t):
    sin_samp.append(sin_t[start:start+frame])
    cos_samp.append(cos_t[start:start+frame])
    start += frame+1
    
## data setup 
seq_len = len(sin_samp) ## 90
samp_size = sin_samp[0].shape[0] ## 10
#print(seq_len, samp_size)

input = torch.from_numpy(np.asarray(sin_samp,dtype=np.float32))
target = torch.from_numpy(np.asarray(cos_samp,dtype=np.float32))

## Initiate and train models
dataset = (input,target)

rnn_model = myRNN(input_shape=1,hidden_size=10,output_size=1)
lstm_model = myLSTM(input_shape=1,hidden_size=10,output_size=1)
gru_model = myGRU(input_shape=1,hidden_size=10,output_size=1)

print('RNN training:')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=rnn_model.parameters(), lr=1e-2)
EPOCHS = 300
rnn_losses = training(epochs=EPOCHS,dataset=dataset,model=rnn_model,loss_fn=loss_fn,optimizer=optimizer)

print('\nLSTM training:')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=lstm_model.parameters(), lr=1e-2)
lstm_losses = training(epochs=EPOCHS,dataset=dataset,model=lstm_model,loss_fn=loss_fn,optimizer=optimizer)

print('\nGRU training:')
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=gru_model.parameters(), lr=1e-2)
gru_losses = training(epochs=EPOCHS,dataset=dataset,model=gru_model,loss_fn=loss_fn,optimizer=optimizer)



plt.figure()
plt.plot(np.linspace(0,EPOCHS,EPOCHS),rnn_losses)
plt.plot(np.linspace(0,EPOCHS,EPOCHS),lstm_losses)
plt.plot(np.linspace(0,EPOCHS,EPOCHS),gru_losses)
plt.title('Training loss per epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.legend(['RNN', "LSTM","GRU"])
plt.show()