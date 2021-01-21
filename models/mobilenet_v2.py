import tensorflow as tf

def mobilenet_v2(img_size, n_classes):
    #Using Mobilenetv2 as the base
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=True, weights=None, classes=n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model