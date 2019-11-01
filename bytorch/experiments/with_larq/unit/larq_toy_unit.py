import numpy as np
import larq as lq
import tensorflow as tf


def build_model():
    model = tf.keras.models.Sequential()

    model.add(
        lq.layers.QuantDense(
            3,
            activation="hard_tanh",
            kernel_quantizer="ste_sign",
            use_bias=False,
            input_shape=(3,),
        )
    )

    model.add(
        lq.layers.QuantDense(
            3,
            activation="hard_tanh",
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            use_bias=False,
        )
    )

    model.add(
        lq.layers.QuantDense(
            3, input_quantizer="ste_sign", kernel_quantizer="ste_sign", use_bias=False
        )
    )

    return model


def main():
    model = build_model()

    model.layers[0].set_weights(
        [np.array([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])]
    )
    model.layers[1].set_weights(
        [np.array([[1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])]
    )
    model.layers[2].set_weights(
        [np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])]
    )

    batch = np.array([[0.1, -0.3, -0.5], [-.1, .3, -0.5], [-1., -0.1, .2]])
    label = np.array([0, 1, 2])

    opt = lq.optimizers.Bop(tf.keras.optimizers.Adam(0.1), threshold=1e-6, gamma=1e-5)
    # opt = tf.keras.optimizers.Adam(0.1)

    model.compile(
        optimizer=opt, loss="categorical_hinge", metrics=["accuracy"]
    )

    model.fit(batch, label, epochs=100, verbose=2)

    l = model.predict(batch)
    print(l)

    e = model.evaluate(batch)
    print(e)

if __name__ == "__main__":
    main()
