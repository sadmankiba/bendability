from __future__ import annotations

from data_organizer import DataOrganizer, ShapeOrganizerFactory, \
        SequenceLibrary, FeatureSelectorFactory, DataOrganizeOptions, \
            TrainTestSequenceLibraries
from constants import CNL, RL, TL

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, LassoCV
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
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
        Runs Scikit-learn classifier to classify C0 value with k-mer count & helical separation.
        """
        X_train, X_test, y_train, y_test = self.organizer.get_seq_train_test(classify=True)

        classifiers = []
        # classifiers.append(('LogisticRegression_C_1', LogisticRegression(C=1)))
        classifiers.append(('LogisticRegression_C_0.1', LogisticRegression(C=0.1)))
        classifiers.append(('SVC', SVC()))
        classifiers.append(('GradientBoost', GradientBoostingClassifier()))
        classifiers.append(('NN', MLPRegressor()))
        # classifiers.append(('KNN', KNeighborsClassifier()))

        # classifiers.append(('rf', RandomForestClassifier(n_estimators=5, max_depth=32))

        result_cols = ['Classifier', 'Test Accuracy', 'Train Accuracy']
        clf_result = pd.DataFrame(columns=result_cols)
        for name, clf in classifiers:
            clf.fit(X_train, y_train)
            clf_result = pd.concat([clf_result, pd.DataFrame([[name, clf.score(X_train, y_train), clf.score(X_test, y_test)]], columns=result_cols)], ignore_index=True)

        clf_result.to_csv(f'data/generated_data/results/classification.tsv', sep='\t', index=False)


    def run_seq_regression(self) -> None:
        """
        Runs Scikit-learn regression models to classify C0 value with k-mer count & helical separation.
        """
        X_train, X_test, y_train, y_test = self.organizer.get_seq_train_test(classify=False)

        regressors = []
        regressors.append(('SVR_C_0.1', SVR(C=0.1)))
        regressors.append(('SVR_C_1', SVR(C=1)))
        #regressors.append(('LinearRegression', LinearRegression()))
        # regressors.append(('Ridge_alpha_1', Ridge(alpha=1)))
        # regressors.append(('Ridge_alpha_5', Ridge(alpha=5)))
        # regressors.append(('Lasso', LassoCV()))
        # regressors.append(('NN', MLPRegressor()))

        result_cols = ['Regression Model', 'Test Accuracy', 'Train Accuracy']
        reg_result = pd.DataFrame(columns=result_cols)
        for name, reg in regressors:
            reg.fit(X_train, y_train)
            reg_result = pd.concat([reg_result, pd.DataFrame([[name, reg.score(X_train, y_train), reg.score(X_test, y_test)]], columns=result_cols)], ignore_index=True)

        reg_result.to_csv(f'data/generated_data/results/regression.tsv', sep='\t', index=False)


    def run_shape_seq_classifier(self) -> None:
        pass


if __name__ == '__main__':
    libraries: TrainTestSequenceLibraries = {
        'train': [TL],
        'test': [RL], 
        'train_test': [],
        'seq_start_pos': 1,
        'seq_end_pos': 50
    }

    # shape_factory = ShapeOrganizerFactory('normal', 'ProT')
    # shape_organizer = shape_factory.make_shape_organizer(library)
    feature_factory = FeatureSelectorFactory('all')
    selector = feature_factory.make_feature_selector()

    options: DataOrganizeOptions = {
        'k_list': [2,3,4],
        'range_split': np.array([0.2, 0.6, 0.2]),
        'binary_class': False,
        'balance': False,
        'c0_scale': 20
    }

    organizer = DataOrganizer(libraries, None, selector, options)

    model = Model(organizer)

    # model.run_seq_classifier()
    model.run_seq_regression()
    # model.run_shape_cnn_classifier()
