import larq as lq
import tensorflow as tf
import tensorflow.keras as tk

from typing import List


def get_cifar_data():
    pass


def build_model(
    use_binary_weights=True, only_train_bm_layers=False, use_bm_layers=False
):
    pass


def train_model(model, epochs=6):
    (train_images, train_labels), _ = get_mnist_data()

    model.fit(train_images, train_labels, batch_size=64, epochs=epochs, verbose=False)


def test_model(model):
    _, (test_images, test_labels) = get_mnist_data()

    test_loss, test_acc = model.evaluate(test_images, test_labels)

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


def main(with_bm=True):
    binary_weight_model: tk.Model = build_model(
        use_binary_weights=True, use_bm_layers=with_bm
    )

    # Train a model with binary weights
    train_model(binary_weight_model, epochs=6)

    # Create an equal model which will use real-valued weights
    real_weight_model: tk.Model = build_model(
        use_binary_weights=False, only_train_bm_layers=True, use_bm_layers=with_bm
    )
    real_weight_model.set_weights(binary_weight_model.get_weights())

    print(
        f"models have same weight: {are_layers_equal(binary_weight_model, real_weight_model, ignore_bm=True)}"
    )

    # Compare accuracies between both models
    binary_model_acc, _ = test_model(binary_weight_model)
    real_model_acc, _ = test_model(real_weight_model)

    # Retrain the batch normalization weights of the real-weighted model
    train_model(real_weight_model, epochs=6)

    print(
        f"models have same weight after retraining: {are_layers_equal(binary_weight_model, real_weight_model, ignore_bm=True)}"
    )

    real_model_retrained_acc, _ = test_model(real_weight_model)

    print(f"binary model accuracy: {binary_model_acc:.2f}")
    print(f"real model accuracy: {real_model_acc:.2f}")
    print(f"real model retrained accuracy: {real_model_retrained_acc:.2f}")


if __name__ == "__main__":
    print("with batch norm\n\n")
    main(with_bm=True)
    print("without batch norm\n")
    main(with_bm=False)
