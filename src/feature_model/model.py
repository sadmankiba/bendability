from __future__ import annotations
from enum import Enum, auto
from datetime import datetime
from pathlib import Path

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
from nptyping import NDArray

from .data_organizer import (
    DataOrganizer,
    TrainTestSequenceLibraries,
    SequenceLibrary,
    DataOrganizeOptions,
    FeatureSelector, 
    ShapeOrganizerFactory
)
from util.constants import RL, TL
from .feat_selector import FeatureSelectorFactory


class Model:
    """
    A class to train model on DNA mechanics libraries.
    """

    @classmethod
    def _get_result_path(self, dir_name: str):
        """
        Makes result path. Creates if needed.
        """
        cur_date = datetime.now().strftime("%Y_%m_%d")
        cur_time = datetime.now().strftime("%H_%M")

        return Path(
            f"data/generated_data/results/{dir_name}/{cur_date}/{cur_time}.tsv"
        ).mkdir(parents=True, exist_ok=True)

    @classmethod
    def run_shape_cnn_classifier(
        self,
        X_train_valid: NDArray,
        X_test: NDArray,
        y_train_valid: NDArray,
        y_test: NDArray,
    ) -> None:
        """
        Run a CNN classifier on DNA shape values.

        Args:
            shape_name: Name of structural feature
            c0_range_split: A numpy 1D array denoting the point of split for classification

        """
        model, history = self._train_shape_cnn_classifier(X_train_valid, y_train_valid)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print("test_acc", test_acc)

        # Plot
        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    def _train_shape_cnn_classifier(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[models.Sequential, tf.keras.callbacks.History]:
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
        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(X.shape[1], kernel_size),
                strides=1,
                activation="relu",
                input_shape=(X.shape[1], X.shape[2], 1),
            )
        )
        model.add(layers.MaxPooling2D((2, 2), padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation="relu"))
        model.add(layers.Dense(np.unique(y).size))

        print(model.summary())

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        history = model.fit(
            X_train, y_train, epochs=15, validation_data=(X_valid, y_valid)
        )

        return model, history

    @classmethod
    def run_seq_classifier(
        self, X_train: NDArray, X_test: NDArray, y_train: NDArray, y_test: NDArray
    ) -> None:
        """
        Runs Scikit-learn classifier to classify C0 value with k-mer count & helical separation.
        """
        classifiers = []
        # classifiers.append(('LogisticRegression_C_1', LogisticRegression(C=1)))
        classifiers.append(("LogisticRegression_C_0.1", LogisticRegression(C=0.1)))
        classifiers.append(("SVC", SVC()))
        classifiers.append(("GradientBoost", GradientBoostingClassifier()))
        classifiers.append(("NN", MLPClassifier()))
        # classifiers.append(('KNN', KNeighborsClassifier()))

        # classifiers.append(('rf', RandomForestClassifier(n_estimators=5, max_depth=32))

        result_cols = ["Classifier", "Test Accuracy", "Train Accuracy"]
        clf_result = pd.DataFrame(columns=result_cols)
        for name, clf in classifiers:
            clf.fit(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            train_acc = clf.score(X_train, y_train)
            print("Model:", name, "Train acc:", train_acc, ", Test acc:", test_acc)
            clf_result = pd.concat(
                [
                    clf_result,
                    pd.DataFrame([[name, test_acc, train_acc]], columns=result_cols),
                ],
                ignore_index=True,
            )
            clf_result.to_csv(
                self._get_result_path(dir_name="classification"), sep="\t", index=False
            )

    @classmethod
    def run_seq_regression(self, X_train, X_test, y_train, y_test) -> None:
        """
        Runs Scikit-learn regression models to classify C0 value with k-mer count & helical separation.
        """
        regressors = []
        regressors.append(("SVR_C_0.1", SVR(C=0.1)))
        regressors.append(("SVR_C_1", SVR(C=1)))
        # regressors.append(('LinearRegression', LinearRegression()))
        # regressors.append(('Ridge_alpha_1', Ridge(alpha=1)))
        # regressors.append(('Ridge_alpha_5', Ridge(alpha=5)))
        # regressors.append(('Lasso', LassoCV()))
        # regressors.append(('NN', MLPRegressor()))

        result_cols = ["Regression Model", "Test Accuracy", "Train Accuracy"]
        reg_result = pd.DataFrame(columns=result_cols)
        for name, reg in regressors:
            reg.fit(X_train, y_train)
            test_acc = reg.score(X_test, y_test)
            train_acc = reg.score(X_train, y_train)
            print("Model:", name, " Train acc:", train_acc, ", Test acc:", test_acc)
            reg_result = pd.concat(
                [
                    reg_result,
                    pd.DataFrame([[name, test_acc, train_acc]], columns=result_cols),
                ],
                ignore_index=True,
            )
            reg_result.to_csv(
                self._get_result_path(dir_name="regression"), sep="\t", index=False
            )

    @classmethod
    def run_shape_seq_classifier(self) -> None:
        pass


class LibrariesParam:
    tl_rl = TrainTestSequenceLibraries(
        train=[SequenceLibrary(name=TL, quantity=20000)],
        test=[SequenceLibrary(name=RL, quantity=5000)],
    )


class ModelCat(Enum):
    CLASSIFIER = auto()
    REGRESSOR = auto()
    SHAPE_CLASSIFIER = auto()


class ModelRunner:
    @classmethod
    def run_model(
        cls,
        libs: TrainTestSequenceLibraries,
        options: DataOrganizeOptions,
        featsel: FeatureSelector,
        cat: ModelCat,
    ):
        shape_organizer = None
        if cat == ModelCat.SHAPE_CLASSIFIER:
            shape_factory = ShapeOrganizerFactory("normal", "ProT")
            shape_organizer = shape_factory.make_shape_organizer(libs)

        organizer = DataOrganizer(libs, shape_organizer, featsel, options)

        if cat == ModelCat.CLASSIFIER:
            Model.run_seq_classifier(*organizer.get_seq_train_test(classify=True))
        elif cat == ModelCat.REGRESSOR:
            Model.run_seq_regression(*organizer.get_seq_train_test(classify=False))
        elif cat == ModelCat.SHAPE_CLASSIFIER:
            Model.run_shape_cnn_classifier(*organizer.get_shape_train_test())
