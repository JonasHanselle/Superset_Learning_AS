import logging
from sklearn.utils import shuffle
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# this should be renamed for final release, GLM stands for generalized loss minimization


class NeuralNetworkGLM:
    def __init__(self, seed, cutoff, loss_function="superset",
                 num_epochs=5000,
                 learning_rate=0.05,
                 batch_size=32,
                 patience=16,
                 es_val_ratio=0.3,
                 early_stop_interval=5,
                 log_losses=True,
                 hidden_layer_sizes=[32],
                 activation_function="sigmoid"):
        self.network = None
        self.logger = logging.getLogger("NeuralNetSuperSetLoss")
        self.loss_history = []
        self.es_val_history = []
        self.seed = seed
        self.cutoff = cutoff
        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.es_val_ratio = es_val_ratio
        self.early_stop_interval = early_stop_interval
        self.log_losses = log_losses
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        K.set_floatx("float64")

    def build_network(self,
                      num_features,
                      hidden_layer_sizes=[32],
                      activation_function="sigmoid"):
        input_layer = keras.layers.Input(num_features, name="input_layer")
        hidden_layers = input_layer
        if hidden_layer_sizes is None:
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
            hidden_layers = keras.layers.Dense(
                num_features, activation=activation_function)(hidden_layers)
        else:
            for layer_size in hidden_layer_sizes:
                hidden_layers = keras.layers.Dense(
                    layer_size, activation=activation_function,)(hidden_layers)

        # Use linear activation function in output layer for regression
        output_layer = keras.layers.Dense(1,
                                          activation="linear",
                                          name="output_layer", use_bias=True)(hidden_layers)
        return keras.Model(inputs=input_layer, outputs=output_layer)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            ):
        """Fit the network to the given data.

        Arguments:
            x {np.ndarray} -- Features
            y {np.ndarray} -- Performances
            regression_loss {String} -- Which regression loss
            should be applied, "Squared" and "Absolute" are
            supported
        """
        print("seed", self.seed)
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        print(self.hidden_layer_sizes)
        num_features = x.shape[1]
        self.network = self.build_network(
            num_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation_function=self.activation_function)

        self.network.make_predict_function()
        self.network.summary()

        self.loss_history = []
        self.es_val_history = []

        X_train, X_val, y_train, y_val = train_test_split(
            x, y, random_state=self.seed, test_size=self.es_val_ratio)

        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        train_data = train_data.batch(self.batch_size)
        val_data = val_data.batch(1)

        def superset_loss(model, x, y, cutoff):
            """Compute generalized superset loss. This loss consists distinguishes three cases. 
            Case 1: Both the prediction and the ground truth are larger than the cutoff, then the loss function is zero.
            Case 2: The ground truth is larger than the cutoff and the prediction is below the cutoff, the squared error between the prediction and the cutoff is computed
            Case 3: if the ground truth is smaller than the cutoff, we compute the squared error between the prediction and the ground truth
            """
            y = y[:, None]
            y_hat = model(x)
            # cases for which y is larger than the cutoff
            # cases for which y_hat is larger than the cutoff
            y_hat_imprecise = y_hat >= cutoff
            y_hat_precise = y_hat < cutoff
            y_precise_ind = y < cutoff
            y_imprecise_ind = y >= cutoff
            zeros = tf.zeros_like(y)
            case_1 = tf.logical_and(y_imprecise_ind, y_hat_imprecise)
            case_2 = tf.logical_and(y_imprecise_ind, y_hat_precise)
            case_3 = y_precise_ind
            err_1 = zeros
            err_2 = tf.where(case_2, tf.square(y_hat - cutoff), zeros)
            err_3 = tf.where(case_3, tf.square(y_hat - y), zeros)
            err_tensor = err_1 + err_2 + err_3
            return tf.reduce_mean(err_tensor)
            # cases for which only y_cut
            # y_hat_minus_c = y_hat - cutoff
            # squared_min_y_hat_minus_c = tf.square(y_hat_minus_c)
            # censored_error = tf.where(y_larger_y_hat_smaller, squared_min_y_hat_minus_c, zeros)
            # only_y_cut = tf.logical_and(
            #     y_cut_ind, tf.logical_not(y_hat_cut_ind))
            # err_2 = tf.where(only_y_cut, tf.square(
            #     y_hat - (cutoff * tf.ones_like(y_hat))), tf.zeros_like(y_hat))
            # # err_2 = y_hat 
            # err_3 = tf.where(tf.logical_not(y_cut_ind), tf.square(
            #     tf.subtract(y, y_hat)), tf.zeros_like(y_hat))
            # # print("err2", err_2)
            # # print("err3", err_3)
            # print("overall err", tf.add(err_2, err_3))
            # err = tf.add(err_2, err_3)

        def l2_loss(model, x, y, cutoff):
            """Compute generalized superset loss. This loss consists distinguishes three cases. 
            Case 1: Both the prediction and the ground truth are larger than the cutoff, then the loss function is zero.
            Case 2: The ground truth is larger than the cutoff and the prediction is below the cutoff, the squared error between the prediction and the cutoff is computed
            Case 3: if the ground truth is smaller than the cutoff, we compute the squared error between the prediction and the ground truth
            """
            y = y[:, None]
            y_hat = model(x)
            return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))

        loss_func = superset_loss
        if self.loss_function == "l2":
            loss_func = l2_loss

        # define gradient of custom loss function
        def grad(model, x, y, cutoff):
            with tf.GradientTape() as tape:
                loss_value = loss_func(
                    model, x, y, cutoff)
            return loss_value, tape.gradient(
                loss_value, model.trainable_weights)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        best_val_loss = float("inf")
        current_best_weights = self.network.get_weights()
        patience_cnt = 0

        for epoch in range(self.num_epochs):
            epoch_reg_loss_avg = tf.keras.metrics.Mean()
            for x, y in train_data:
                loss_value, grads = grad(
                    self.network, x, y, self.cutoff)
                optimizer.apply_gradients(
                    zip(grads, self.network.trainable_weights))
                epoch_reg_loss_avg(loss_value)
            if self.log_losses:
                self.loss_history.append([
                    float(epoch_reg_loss_avg.result()),
                ])

            if epoch % self.early_stop_interval == 0:
                losses = []
                for x, y in val_data:
                    losses.append(
                        loss_func(self.network, x, y, self.cutoff))
                loss_tensor = np.average(losses)
                current_val_loss = tf.reduce_mean(loss_tensor)
                # print("cur val loss", current_val_loss)
                self.es_val_history.append(current_val_loss)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    current_best_weights = self.network.get_weights()
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    # print("patience counter", patience_cnt)
                if patience_cnt >= self.patience:
                    # print("early stopping")
                    break

        # self.network.set_weights(current_best_weights)

    def get_weights(self):
        return self.network.get_weights()

    def predict(self, features: np.ndarray):
        predictions = self.network(features)
        return predictions.numpy()
