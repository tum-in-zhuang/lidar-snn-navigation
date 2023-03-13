import tensorflow as tf
import tensorflow_addons as tfa

bias = False


def create_LoDNN_model(input_shape):
    inp = x = tf.keras.Input(input_shape)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=1,
        name="conv1",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU1")(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=1,
        name="conv2",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU2")(x)

    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), name="avg_pool3"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv4",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU4")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv5",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU5")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv6",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU6")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv7",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU7")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv8",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU8")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv9",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU9")(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=2,
        name="dilated_conv10",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU10")(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=1,
        strides=1,
        padding="same",
        dilation_rate=1,
        name="conv11",
        use_bias=bias,
    )(x)

    x = tf.keras.layers.AveragePooling2D(
        pool_size=(4, 4), strides=(4, 4), name="avg_pool3_2"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding="same",
        dilation_rate=1,
        name="conv13",
        use_bias=bias,
    )(x)
    x = tf.keras.layers.ReLU(name="ReLU13")(x)

    x = tf.keras.layers.Activation("sigmoid", name="sigmoid15")(x)

    model = tf.keras.Model(inp, x)
    return model
