from glob import glob
import os
import librosa
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random

def data_parser(directory):
    '''
    Return waves, ids and digits list, given .wav files location.
    '''
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split('/')[3].split('.')[0] for f in files]
    # extract ids from filename
    ids = [int(f[-2]+f[-1]) if f[-2].isdigit() else int(f[-1]) for f in fnames]
    # extract float arrays of signals
    waves = [librosa.load(fpath,sr=None)[0] for fpath in files]
    # extract digits from filenames
    digits = []
    for i in range(len(fnames)):
        if str(ids[i]) in fnames[i]:
            fnames[i] = fnames[i].replace(str(ids[i]),'')
    
    return waves, ids, fnames


def extract_mfcc(waves, num_mfcc=13, window=25, step=10):
    '''
    Returns 3 lists of np.arrays of shape (frames of signal x num_mfcc)
    '''
    w = window*16 ## length of window in samples
    s = step*16  ## length of step in samples
    mfccs = [
        librosa.feature.mfcc(y=wav, n_mfcc=num_mfcc, n_fft=w, hop_length=s).T ## by default returns (MFCC x Frames) shape
        for wav in tqdm(waves, desc='Extracting MFCC features...')
    ]
    deltas = [librosa.feature.delta(mfcc) for mfcc in mfccs]
    ddeltas = [librosa.feature.delta(d) for d in deltas]
    
    print(f'Feature extraction completed: {num_mfcc} MFCCs per frame.')
    return mfccs, deltas, ddeltas


def extract_mfsc(waves, num_mfsc=13, window=25, step=10):
    '''
    Returns list of np.arrays of shape (frames of signal x num_mfsc)
    '''
    w = window*16 ## length of window in samples
    s = step*16  ## length of step in samples
    mfscs = [
        librosa.feature.melspectrogram(y=wav, n_mels=num_mfsc, n_fft=w, hop_length=s).T ## by default returns (MFSC x Frames) shape
        for wav in tqdm(waves, desc='Extracting MFSC features...')
    ]
    
    print(f'Feature extraction completed: {num_mfsc} MFSCs per frame.')
    return mfscs


def find_indices(lista,item):
    indices = []
    for idx, value in enumerate(lista):
        if value == item:
            indices.append(idx)
    return indices

def digitsHistogram(n1,n2,mfccs,digits):
    ### find first 2 MFCCs of n1 digit for all speakers
    n1_indeces = find_indices(digits,n1)
    n1_feats0 = np.array([])
    n1_feats1 = np.array([])
    for i in n1_indeces:
        n1_feats0 = np.concatenate((n1_feats0,mfccs[i][:,0]),axis=0)
        n1_feats1 = np.concatenate((n1_feats1,mfccs[i][:,1]),axis=0)
    
    ### find first 2 MFCCs of n2 digit for all speakers
    n2_indeces = find_indices(digits,n2)
    n2_feats0 = np.array([])
    n2_feats1 = np.array([])
    for i in n2_indeces:
        n2_feats0 = np.concatenate((n2_feats0,mfccs[i][:,0]),axis=0)
        n2_feats1 = np.concatenate((n2_feats1,mfccs[i][:,1]),axis=0)
    
    plt.hist(n1_feats0)
    plt.title(f'Histogram of 1st coeff for digit {n1}')
    plt.show()
    plt.hist(n1_feats1)
    plt.title(f'Histogram of 2nd coeff for digit {n1}')
    plt.show()
    plt.hist(n2_feats0)
    plt.title(f'Histogram of 1st coeff for digit {n2}')
    plt.show()
    plt.hist(n2_feats1)
    plt.title(f'Histogram of 2nd coeff for digit {n2}')
    plt.show()
    
    print(n1_feats1.shape)

def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='upper left')


def createMfccDataset(data,labels):
    '''
    Returns list of tuples: (label, feature vector with this label)
    
    Each feature vector is a row from a feature table in data.
    THE IDEA: each vector of each MFCC table is a feature vector that belongs to a certain class
    '''
    res = []
    for idx,vec in enumerate(data):
            res.append((labels[idx],vec))
    return res
    
def createFeatsDataset(data,labels):
    res = []
    for idx,vec in enumerate(data):
        res.append((labels[idx],normalize([vec])[0]))
    return res
    

def split_data(dataset,train_perc = 0.7, test_perc = 0.3):
    '''
    Input: dataset (list of tuples) , train percentage, test percentage
    Output: train data,train labels,test data, test labels
    '''
    if train_perc+test_perc != 1:
        print("ERROR:Train and test percantages must add to 1.")
        return
    
    ## select train_perc*all data random ids 
    train_ids = random.sample(range(0,len(dataset)-1),int(train_perc*len(dataset)))
    
    ## split data according to random ids
    x_train = [dataset[i][1] for i in train_ids]
    y_train = [dataset[i][0] for i in train_ids]
    
    x_test = [dataset[i][1] for i in range(len(dataset)-1) if i not in train_ids]
    y_test = [dataset[i][0] for i in range(len(dataset)-1) if i not in train_ids]
    
    return x_train, y_train, x_test, y_test
    
    





