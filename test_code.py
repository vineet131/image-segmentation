import tensorflow as tf

def preview_record(raw_image_dataset):
    """
    Preview a single image from the TFRecord file
    """
    """for raw_record in raw_image_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
        cv2.imshow("image", cv2.imdecode(example, cv2.IMREAD_COLOR))
        cv2.waitKey(0)"""
    
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    for image_features in parsed_image_dataset:
        image_raw = image_features['image/encoded'].numpy()
        class_raw = image_features['image/segmentation/class/encoded'].numpy()
        img_np = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        class_np = cv2.imdecode(np.frombuffer(class_raw, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("image", img_np)
        cv2.imshow("class", class_np)
        cv2.waitKey(0)
        break

def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary.
    image_feature_description = {'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/segmentation/class/format': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)