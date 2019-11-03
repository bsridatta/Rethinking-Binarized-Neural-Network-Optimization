import larq as lq
import tensorflow as tf


def get_mnist_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to be between -1 and 1
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    kwargs = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        # kernel_constraint="weight_clip",
    )

    model = tf.keras.models.Sequential()

    model.add(
        lq.layers.QuantConv2D(
            32,
            (3, 3),
            kernel_quantizer="ste_sign",
            # kernel_constraint="weight_clip",
            use_bias=False,
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(tf.keras.layers.Flatten())

    model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Activation("softmax"))

    return model


def main():
    (train_images, train_labels), (test_images, test_labels) = get_mnist_data()

    print(type(train_images))
    print(train_images.shape)

    print(type(train_labels))
    print(train_labels.shape)

    model = build_model()

    lq.models.summary(model)

    opt = lq.optimizers.Bop(tf.keras.optimizers.Adam(), threshold=1e-5, gamma=1e-3)

    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, batch_size=64, epochs=6)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(f"Test accuracy {test_acc * 100:.2f} %")


if __name__ == "__main__":
    main()
