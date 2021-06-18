from read_data import get_processed_data
from shape import find_valid_cols, run_dna_shape_r_wrapper

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import math
from pathlib import Path
import random

# Two types of shape feature processing 
#   1. Only discrete values
#   2. One-hot encode discrete values

def encode_shape(shape_arr, shape_str, n_split):
    # possib_shape_df = pd.read_csv(f'data/generated_data/{shape_arr}_possib_values.tsv', sep='\t')
    # shape_min = min(possib_shape_df[shape_str])
    # shape_max = max(possib_shape_df[shape_str])
    shape_min = np.min(shape_arr) 
    shape_max = np.max(shape_arr)

    shape_range = ((shape_max - shape_min) / n_split) + 0.001
    return np.floor((shape_arr - shape_min) / shape_range) 


def one_hot_encode_shape(shape_arr, shape_str, n_split):
    """
    Transform shape values to discrete numbers and perform One-hot encoding

    args:
        shape_arr (numpy.Ndarray): shape values of DNA sequences
        shape_str: a str from ['EP', 'HelT', 'MGW', 'ProT', 'Roll'] 
        n_split: number of discrete values
    
    returns:
        a numpy Ndarray containing one-hot encoded shape values of sequences 
    """
    encoded_shape_arr = encode_shape(shape_arr, shape_str, n_split)
    ohe = OneHotEncoder()
    ohe.fit(np.array(list(range(n_split))).reshape(-1,1))

    arr_3d = None
    for i in range(encoded_shape_arr.shape[0]):
        seq_enc_shape_arr = ohe.transform(encoded_shape_arr[i].reshape(-1,1)).toarray().transpose()
        seq_enc_shape_arr = seq_enc_shape_arr.reshape(1, seq_enc_shape_arr.shape[0], seq_enc_shape_arr.shape[1])
        if i == 0:
            arr_3d = seq_enc_shape_arr
        else:
            arr_3d = np.concatenate((arr_3d, seq_enc_shape_arr), axis=0)

    return arr_3d


def classify_arr(arr, range_split):
    """
    Encodes a 1D numpy array into discrete values 

    args:
        arr: numpy array
        range_split: a numpy array of elements from 0 to 1 that sums to 1. eg. [0.3, 0.8, 1.0]
    
    returns:
        arr classified into integer between 0 to range_split.size - 1 
    """
     
    range_arr = np.array(sorted(arr))[np.floor(arr.size * range_split - 1).astype(int)]     # ranges for array according to range_split

    return np.array([ np.searchsorted(range_arr, e) for e in arr ]) 
        

def run_shape_cnn(X, y):
    X = X.reshape(list(X.shape) + [1])

    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(X.shape[1], 4), strides=1,
        activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(np.unique(y).size))

    print(model.summary())

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.15)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1)

    history = model.fit(X_train, y_train, epochs=10, 
                        validation_data=(X_valid, y_valid))
    
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    print('test_acc', test_acc)
    

if __name__ == '__main__':
    (cnl_df, rl_df, tl_df, chrvl_df, libl_df) = get_processed_data()
    library_name = 'rl'
    sequences = rl_df['Sequence']
    
    shape_name = 'ProT'
    shape_arr = run_dna_shape_r_wrapper(rl_df, False)[shape_name] 
    
    # Prune not-nan columns 
    valid_cols = find_valid_cols(shape_arr[0].flatten())
    shape_arr = shape_arr[:, valid_cols]
    print('shape_arr', shape_arr.shape)

    num_shape_encode = 12 # CHANGE HERE
    

    saved_shape_arr_file = Path(f"data/generated_data/saved_arrays/{shape_name}_ohe_{library_name}.npy")
    if saved_shape_arr_file.is_file():
        # file exists
        with open(saved_shape_arr_file, 'rb') as f:
            X = np.load(f)
    else: 
        X = one_hot_encode_shape(shape_arr, 'ProT', num_shape_encode)
        with open(saved_shape_arr_file, 'wb') as f:
            np.save(f, X)
        
    print('X', X.shape)
    print(X[random.sample(range(X.shape[0]), 5)])
    assert X.shape == (shape_arr.shape[0], num_shape_encode, shape_arr.shape[1])

    c0_range_split = np.array([0.2, 0.8, 1.0]) # CHANGE HERE 
    y = classify_arr(rl_df['C0'].to_numpy(), c0_range_split)
    
    run_shape_cnn(X, y)