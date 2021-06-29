from __future__ import annotations

from data_organizer import DataOrganizer, ShapeOrganizerFactory, \
        SequenceLibrary, FeatureSelectorFactory

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class Model:
    """
    A class to train model on DNA mechanics libraries.
    """
    def __init__(self, organizer: DataOrganizer):
        """
        Constructor

        Args:
            organizer: Data organizer object
        """
        self.organizer = organizer
        

    def _train_shape_cnn_classifier(self, X: np.ndarray, y: np.ndarray) \
        -> tuple[models.Sequential, tf.keras.callbacks.History]:
        """
        Train a CNN model on data

        Args:
            X: training and validation samples
            y: training and validation targets

        Returns:
            A tuple containing
                * model: a tensorflow sequential model
                * history: history object of training
        """
        kernel_size = 8
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(X.shape[1], kernel_size), strides=1,
            activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(np.unique(y).size))

        print(model.summary())

        model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        history = model.fit(X_train, y_train, epochs=15,
                            validation_data=(X_valid, y_valid))

        return model, history


    def run_shape_cnn_classifier(self, shape_name: str, c0_range_split: np.ndarray, \
            encode: bool, binary: bool) -> None:
        """
        Run a CNN classifier on DNA shape values.

        Args:
            shape_name: Name of structural feature
            c0_range_split: A numpy 1D array denoting the point of split for classification
            
        """
        # Train
        X_train_valid, X_test, y_train_valid, y_test = self.organizer.get_shape_train_test() 
        model, history = self._train_shape_cnn_classifier(X_train_valid, y_train_valid)
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
        print('test_acc', test_acc)

        # Plot
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()


    def run_seq_classifier(self) -> None:
        """
        Runs Scikit-learn classifier to classify C0 value with k-mer count.
        """
        X_train, X_test, y_train, y_test = self.organizer.get_kmer_train_test()
        
        forest = RandomForestClassifier(n_estimators=5)
        forest.fit(X_train, y_train)

        print('train accuracy:', forest.score(X_train, y_train))
        print('test accuracy:', forest.score(X_test, y_test))
        
            
    def run_shape_seq_classifier(self) -> None:
        pass


if __name__ == '__main__':
    library: SequenceLibrary = {
        'name': 'cnl', 
        'seq_start_pos': 1,
        'seq_end_pos': 50
    }

    shape_factory = ShapeOrganizerFactory('normal', 'ProT')
    shape_organizer = shape_factory.make_shape_organizer(library)
    feature_factory = FeatureSelectorFactory('manual')
    selector = feature_factory.make_feature_selector()
    
    organizer = DataOrganizer(library, shape_organizer, selector, k_list=[2, 3, 4], 
        range_split=np.array([0.2, 0.6, 0.2]),  binary_class=False)
    
    model = Model(organizer)
    
    model.run_seq_classifier()
    # model.run_shape_cnn_classifier()
