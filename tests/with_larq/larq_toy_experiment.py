import numpy as np
import larq as lq
import tensorflow as tf


def generate_toy_data(n_samples, n_features):
    data = []
    labels = []

    for idx, (mean, var) in enumerate([(0.8, 0.001), (0, 0.001), (-0.8, 0.001)]):
        for _ in range(n_samples):
            sample = np.random.normal(mean, var, n_features)
            label = idx

            data.append(sample)
            labels.append(label)

    return np.array(data), np.array(labels)


def get_toy_data(normalize=False):
    train_data, train_labels = generate_toy_data(1024, 100)
    test_data, test_labels = generate_toy_data(100, 100)

    if normalize:
        pass

    return (train_data, train_labels), (test_data, test_labels)


def build_model():
    model = tf.keras.models.Sequential()

    model.add(
        lq.layers.QuantDense(
            50,
            # input_quantizer="ste_sign",
            activation='hard_tanh',
            use_bias=False, input_shape=(100,)
        )
    )

    model.add(
        lq.layers.QuantDense(
            25,
            activation='hard_tanh',
            use_bias=False
        )
    )

    model.add(
        lq.layers.QuantDense(
            3,
            use_bias=False
        )
    )

    model.add(tf.keras.layers.Activation("softmax"))

    return model


def main():
    (train_images, train_labels), (test_images, test_labels) = get_toy_data()

    for _ in range(0, 10):
        model = build_model()

        opt = lq.optimizers.Bop(tf.keras.optimizers.Adam(0.01), threshold=1e-6, gamma=1e-5)

        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        model.fit(train_images, train_labels, batch_size=16, epochs=10, shuffle=True, verbose=False)

        train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=False)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=False)

        print(
            f"train accuracy: {train_acc: .3f} test accuracy: {test_acc: .3f}"
        )


if __name__ == "__main__":
    main()
