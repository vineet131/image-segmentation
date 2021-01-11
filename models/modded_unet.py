import tensorflow as tf, numpy as np

def modded_unet(img_size, n_classes):
    #Using Mobilenetv2 as the base
    model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False)
    #Encoder
    layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
    ]
    layers = [model.get_layer(name).output for name in layer_names]

    #Feature extraction
    down_stack = tf.keras.Model(inputs=model.input, outputs=layers)
    down_stack.trainable = False

    #Upsampling method
    def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

    #Decoder
    up_stack = [
            upsample(512, 3),  # 4x4 -> 8x8
            upsample(256, 3),  # 8x8 -> 16x16
            upsample(128, 3),  # 16x16 -> 32x32
            upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="img_input")
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(filters=n_classes, kernel_size=3, strides=2, padding='same')  #64x64 -> 128x128

    x = last(x)

    model_final = tf.keras.Model(inputs=inputs, outputs=x)
    model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=["accuracy"],
                            run_eagerly=True)
    print(model_final.summary())
    return model_final