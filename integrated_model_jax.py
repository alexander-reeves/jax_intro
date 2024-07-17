is_jax = True
import flax
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import pickle
from flax import linen as nn
from jax.nn import sigmoid
from flax import traverse_util
import jax.random as random
from jax import jit, vmap, disable_jit
import jax.numpy as numpy
from jax.numpy import (
        isnan,
        isinf,
        savez,
        load,
        array,
        ndarray,
        conj,
        ones,
        tan,
        log,
        logspace,
        swapaxes,
        empty,
        linspace,
        arange,
        delete,
        where,
        pi,
        cos,
        sin,
        log,
        exp,
        sqrt,
        concatenate,
        linalg,
        eye,
        einsum,
        einsum_path,
        zeros,
        sum,
        pad,
        diag,
        block,
        array_equal,
        meshgrid,
        geomspace,
        moveaxis,
        ones_like,
        empty_like,
        real,
        zeros_like,
        float32,
        float64,
        dot,
        multiply,
        add,
        subtract,
        unique,
        hstack,
        isin,
        newaxis,
        ix_,
        transpose,
        interp,
        moveaxis,
        rollaxis,
        logical_or,
        inf,
        tanh,
        column_stack,
        power,
        asarray,
        sign,
        matmul,
        logical_and,
        column_stack,
        all,
        mean,
        ndim,
        ceil,
        argwhere,
        cov,
        nansum,
        nanmax,
        identity,
        triu_indices,
        repeat,
        bincount)
from jax import vmap 
from jax.numpy import max as module_max
from jax.numpy import min as module_min
from jax.numpy.fft import rfft
from jax.scipy.linalg import block_diag
from jax.scipy.integrate import trapezoid as trapz
import jax.numpy as jnp

class MyNetwork(nn.Module):
    weights: list  # List of (weight, bias) tuples for each layer
    hyper_params: list  # List of (alpha, beta) for each custom activation layer

    @nn.compact
    def __call__(self, x):
        # Loop over layers except the last one
        for (w, b), (a, b_hyper) in zip(self.weights[:-1], self.hyper_params):
            x = CustomActivation_jax(a=a, b=b_hyper)(dot(x, w) + b)

        # Final layer (no activation)
        final_w, final_b = self.weights[-1]
        x = dot(x, final_w) + final_b
        return x


class CustomActivation_jax(nn.Module):
    a: float  # alpha hyperparameter
    b: float  # beta hyperparameter

    @nn.compact
    def __call__(self, x):
        return multiply(
            add(self.b, multiply(sigmoid(multiply(self.a, x)), subtract(1.0, self.b))),
            x,
        )


# Custom activation function to match https://arxiv.org/abs/2106.03846 for original tf model
class CustomActivation(Layer):
    def build(self, input_shape):
        # Trainable weight variables for alpha and beta initialized with random normal distribution
        self.alpha = self.add_weight(
            name="alpha",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        super(CustomActivation, self).build(input_shape)

    def call(self, inputs):
        return (self.beta + tf.sigmoid(self.alpha * inputs) * (1 - self.beta)) * inputs


# Custom loss function for original tf model
class CustomLoss(Layer):
    def __init__(self, element_weights=None, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
        self.element_weights = (
            tf.convert_to_tensor(element_weights, dtype=tf.float32)
            if element_weights is not None
            else tf.constant(1.0, dtype=tf.float32)
        )

    def call(self, y_true, y_pred):
        # Compute element-wise squared errors
        squared_errors = tf.square(y_true - y_pred)

        if self.element_weights is not None:
            # Apply element-wise weights
            squared_errors = squared_errors / self.element_weights

        # Compute the mean of squared errors
        return tf.reduce_mean(squared_errors)

    def get_config(self):
        config = super(CustomLoss, self).get_config()
        config.update({"element_weights": self.element_weights.numpy()})
        return config


def insert_zero_columns(prediction, zero_columns_indices):
    # Total number of columns in the final output
    num_columns = prediction.shape[1] + len(zero_columns_indices)

    # Initialize the final_prediction array
    final_prediction = zeros((prediction.shape[0], num_columns))

    # Indices in final_prediction where values from prediction will be inserted
    non_zero_indices = delete(arange(num_columns), zero_columns_indices)

    # Insert values from prediction into the correct positions in final_prediction
    final_prediction = final_prediction.at[:, non_zero_indices].set(prediction)

    return final_prediction


class IntegratedModel:
    def __init__(
        self,
        keras_model,
        input_scaler,
        output_scaler,
        parameter_names,
        temp_file=None,
        offset=None,
        log_preprocess=False,
        rescaling_factor=None,
        pca=None,
        pca_scaler=None,
    ):
        self.model = keras_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.offset = offset
        self.rescaling_factor = rescaling_factor
        self.temp_file = temp_file
        self.train_losses = []
        self.val_losses = []
        self.log_preprocess = log_preprocess
        self.parameter_names = parameter_names
        self.pca = pca
        self.pca_scaler = pca_scaler

    def predict(self, data_dict):
        if self.parameter_names is not None:
            required_keys = set(sorted(data_dict.keys()))
            expected_keys = set(self.parameter_names)
            if not expected_keys.issubset(required_keys):
                raise ValueError(
                    f"Input dictionary keys {required_keys} do not match expected keys {expected_keys}"
                )

        # Map parameter values to the correct positions in the input matrix
        num_samples = len(
            next(iter(data_dict.values()))
        )  # Assuming all values in data_dict have the same length
        num_features = len(self.parameter_names)

        data = empty((num_samples, num_features))

        for i, key in enumerate(self.parameter_names):
            data = data.at[:, i].set(data_dict[key])

            # print(data)
            # print(self.parameter_names)
            # print(self.scaler_mean_in)
        scaled_data = (data - self.scaler_mean_in) / self.scaler_scale_in

        # Convert to TensorFlow tensor
        scaled_data = array(scaled_data, dtype=float32)
        prediction = self.jax_model.apply(self.jax_params, scaled_data)
        # Apply PCA transform if PCA is used
        if self.pca is not None:
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out

            # invert the PCA using the mean and mcomponents
            prediction = dot(prediction, self.pca_components) + self.pca_mean
            # prediction = self.pca.inverse_transform(prediction)
            # prediction = self.pca_scaler.inverse_transform(prediction)
            prediction = prediction * self.pca_scaler_scale + self.pca_scaler_mean
        else:
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out

        # Inverse offset and log
        if self.log_preprocess:
            prediction = exp(prediction) + 2 * self.offset

        return prediction

    def restore(self, filename):
        """
        Load pre-saved IntegratedModel attributes'

        Transform tf weights to jax weights dynamically and initialize the jax model

        Parameters:
            filename (str): filename tag (without suffix) where model was saved
        """
        # Load the Keras model from the .h5 format
        tf_model = load_model(
            filename + "_model.h5",
            custom_objects={
                "CustomActivation": CustomActivation,
                "CustomLoss": CustomLoss,
            },
            compile=False,
        )

        self.verbose = True

        tf_params = []
        for layer in tf_model.layers:
            weights = layer.get_weights()
            if weights:  # Check if the layer has parameters
                tf_params.append(weights)

        # load in the weights and hyperparameters
        tf_weights = []
        tf_hyperparams = []
        for i, layer in enumerate(tf_params):
            if i % 2 == 0:
                tf_weights.append(tf_params[i])

            else:
                tf_hyperparams.append(tf_params[i])

        # Load the remaining attributes
        with open(filename + ".pkl", "rb") as f:
            attributes = pickle.load(f)

            try:
                (
                    self.input_scaler,
                    self.output_scaler,
                    self.offset,
                    self.log_preprocess,
                    self.pca,
                    self.pca_scaler,
                    self.parameter_names,
                    self.rescaling_factor,
                ) = attributes

            except:
                print("old model")
                (
                    self.input_scaler,
                    self.output_scaler,
                    self.offset,
                    self.log_preprocess,
                    self.pca,
                    self.parameter_names,
                    self.rescaling_factor,
                ) = attributes

        self.scaler_mean_in = self.input_scaler.mean_  # Mean of the features
        self.scaler_scale_in = (
            self.input_scaler.scale_
        )  # Standard deviation of the features

        print("scaler mean shape", self.scaler_mean_in.shape)

        self.scaler_scale_out = (
            self.output_scaler.scale_
        )  # Standard deviation of the features
        self.scaler_mean_out = (
            self.output_scaler.mean_
        )  # Standard deviation of the features

        print("scaler mean out shape", self.scaler_mean_out.shape)

        try:
            self.pca_scaler_mean = self.pca_scaler.mean_
            self.pca_scaler_scale = self.pca_scaler.scale_

        except:
            if self.verbose:
                print("no pca")

        try:
            # get the parts require to perform the PCA invese transform
            self.pca_mean = self.pca.mean_
            self.pca_components = self.pca.components_

        except:
            if self.verbose:
                print("no pca")

        self.jax_model = MyNetwork(hyper_params=tf_hyperparams, weights=tf_weights)

        # Initialize the model with dummy data
        input_shape = tf_weights[0][0].shape[0]  # the input shape of the first layer
        rng = random.PRNGKey(0)
        dummy_input = ones((1, input_shape))
        params = self.jax_model.init(rng, dummy_input)

        # Flatten the parameters dictionary
        flattened_params = traverse_util.flatten_dict(params)

        # Replace the initialized parameters with your pre-trained weights and hyperparameters
        for i, layer_path in enumerate(flattened_params.keys()):
            if (
                "CustomActivation" in layer_path
            ):  # Assuming this is the name of your custom activation layers
                # Replace hyperparameters
                flattened_params[layer_path] = tf_hyperparams[i]
            else:
                # Replace weights and biases
                flattened_params[layer_path] = tf_weights[i]

        # Unflatten the parameters dictionary
        self.jax_params = traverse_util.unflatten_dict(flattened_params)

        if self.verbose:
            print("restore successful")
