import larq as lq

import tensorflow as tf
import tensorflow.keras as tk

from typing import List
from argparse import ArgumentParser
from time import time
import json
import numpy as np
import os


def get_cifar_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tk.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between -1 and 1
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

    return (train_images, train_labels), (test_images, test_labels)

def build_model(
    use_binary_weights=True, only_train_bm_layers=False, use_bm_layers=False,
    initialization="glorot_uniform", lr=0.01, optimizer="Adam"
):
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign" if use_binary_weights else None,
        kernel_constraint="weight_clip",
        trainable=not only_train_bm_layers,
        kernel_initializer=initialization
    )
    
    model = tk.models.Sequential()

    model.add(
        lq.layers.QuantConv2D(
            32,
            (3, 3),
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
            input_shape=(32, 32, 3),
            trainable=not only_train_bm_layers,
            kernel_initializer=initialization
        )
    )
    model.add(tk.layers.MaxPooling2D((2, 2)))
    model.add(tk.layers.BatchNormalization(
        scale=False)) if use_bm_layers else None

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tk.layers.MaxPooling2D((2, 2)))
    model.add(tk.layers.BatchNormalization(
        scale=False)) if use_bm_layers else None

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tk.layers.BatchNormalization(
        scale=False)) if use_bm_layers else None

    model.add(tk.layers.Flatten())

    model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
    model.add(tk.layers.BatchNormalization(
        scale=False)) if use_bm_layers else None
    model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
    model.add(tk.layers.BatchNormalization(
        scale=False)) if use_bm_layers else None
    model.add(tk.layers.Activation("softmax"))
    if optimizer == "Adam":
        opt = tk.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        opt = tk.optimizers.SGD(learning_rate=lr)
        
    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(model, epochs=4):
    (train_images, train_labels), _ = get_cifar_data()
    tb = tk.callbacks.TensorBoard(
        log_dir=f"./theorem/exper_{hparams.init[0:2]} \
                                 _{hparams.optim[0:2]} \
                                 _{hparams.lr}",
        histogram_freq=0,
        write_graph=True,
    )

    model.fit(
        train_images,
        train_labels,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        callbacks=[tb],
    )


def test_model(model):
    _, (test_images, test_labels) = get_cifar_data()
    test_loss, test_acc = model.evaluate(
        test_images, test_labels, verbose=False)

    return test_acc, test_loss


def are_layers_equal(m1: tk.Model, m2: tk.Model, ignore_bm=False):
    m1_layers: List[tk.layers.Layer] = m1.layers
    m2_layers: List[tk.layers.Layer] = m2.layers

    if len(m1_layers) is not len(m2_layers):
        print("models layer length not equal")
        return False

    for l1, l2 in zip(m1_layers, m2_layers):
        if len(l1.weights) is not len(l2.weights):
            print("model weight length not equal")
            return False

        if "batch_normalization" in l1.name and ignore_bm:
            continue

        for w1, w2 in zip(l1.weights, l2.weights):
            if not tf.reduce_all(w1 == w2):
                print("weights not equal")
                return False

    return True


if __name__ == "__main__":
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--init", type=str, default="random_uniform")
    
    hparams = parser.parse_args()
    
    binary_weight_model: tk.Model = build_model(
        use_binary_weights=True,
        use_bm_layers=True,
        initialization=hparams.init,
        lr=hparams.lr,
        optimizer=hparams.optim
    )

    if hparams.init == "scaled_glorot_uniform":
        for layer in binary_weight_model.layers: 
            weights = layer.get_weights()
            new_weights = []
            for weight in weights:
                weight = np.multiply(weight, 0.01)
                new_weights.append(weight)            
            layer.set_weights(new_weights)
    # Train a model with binary weights
    train_model(binary_weight_model, epochs=hparams.epochs)
    binary_model_acc, _ = test_model(binary_weight_model)

    