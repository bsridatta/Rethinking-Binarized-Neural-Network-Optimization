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
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
    )

    model = tf.keras.models.Sequential()

    model.add(
        lq.layers.QuantDense(
            50,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False, input_shape=(100,)
        )
    )

    model.add(
        lq.layers.QuantDense(
            25,
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        )
    )

    model.add(
        lq.layers.QuantDense(
            3,
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False
        )
    )

    model.add(tf.keras.layers.Activation("softmax"))

    return model


def main():
    (train_images, train_labels), (test_images, test_labels) = get_toy_data()

    model = build_model()

    print(type(train_images))
    print(train_images.shape)

    print(type(train_labels))
    print(train_labels.shape)

    lq.models.summary(model)

    opt = lq.optimizers.Bop(tf.keras.optimizers.Adam(0.01), threshold=1e-3, gamma=1e-3)

    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, batch_size=16, epochs=100, shuffle=True)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(f"Test accuracy {test_acc * 100:.2f} %")


if __name__ == "__main__":
    main()
