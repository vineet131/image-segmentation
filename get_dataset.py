import tensorflow as tf
import tensorflow_datasets as tfds

def import_pascalvoc(img_size):
    """
    Imports the PASCAL VOC TFRecord file. Parameters stored in the TFRecord:
    image/encoded: encoded image content.
    image/filename: image filename.
    image/format: image file format.
    image/height: image height.
    image/width: image width.
    image/channels: image channels.
    image/segmentation/class/encoded: encoded semantic segmentation content.
    image/segmentation/class/format: semantic segmentation file format.

    Used the scripts from TF Deeplab to build this TFRecord, especially the file build_voc2012_data.py

    PASCAL VOC has 20 classes. The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations.
    
    Classes are as follows:
    
    Person: person
    Animal: bird, cat, cow, dog, horse, sheep
    Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
    Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

    Pixel value 0 is background and pixel value 255 (arbitrary boundary b/w background and object class) has been converted to 0
    """
    TRAIN_FILEPATH = "./datasets/train-0000*-of-00004.tfrecord"
    TEST_FILEPATH = "./datasets/trainval-0000*-of-00004.tfrecord"
    VAL_FILEPATH = "./datasets/val-0000*-of-00004.tfrecord"

    train_filename = tf.io.gfile.glob(TRAIN_FILEPATH)
    test_filename = tf.io.gfile.glob(TEST_FILEPATH)
    val_filename = tf.io.gfile.glob(VAL_FILEPATH)
    
    train_dataset = tf.data.TFRecordDataset(train_filename)
    test_dataset = tf.data.TFRecordDataset(test_filename)
    val_dataset = tf.data.TFRecordDataset(val_filename)
    #dataset = dataset.with_options(tf.data.Options())
    #dataset.interleave()

    def process_record(datapoint):
        content_dict = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string),
        }
        parsed = tf.io.parse_example(datapoint, content_dict)
        data, label = parsed['image/encoded'], parsed['image/segmentation/class/encoded']
        
        data = tf.image.decode_jpeg(data, channels=3)
        data = tf.image.resize(data, [img_size, img_size])

        label = tf.image.decode_png(label)
        #Resizing with nearest neighbours maintains the grayscale pixel value corresponding to the label class
        label = tf.cast(tf.image.resize(label, [img_size, img_size], method='nearest'), tf.int32)
        label = tf.where(tf.equal(label, 255), 0, tf.cast(label, tf.int32))
        #label = tf.image.resize(label, [img_size, img_size])

        if tf.random.uniform(()) > 0.5:
            data = tf.image.flip_left_right(data)
            label = tf.image.flip_left_right(label)
        
        data = tf.cast(data, tf.float32) / 255.0
        #label -= 1
        return data, label
    
    return train_dataset.map(process_record), val_dataset.map(process_record), test_dataset.map(process_record)

def import_oxfordpets(img_size):
    """
    Imports the Oxford/IIIT Pets data file.

    This contains 3 main classes: 1 for cat, 2 for dog and 3 for background
    """
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask
    
    def load_image_train(datapoint):
        input_image = tf.image.resize(datapoint['image'], (img_size, img_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_size, img_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        return normalize(input_image, input_mask)
    
    def load_image_test(datapoint):
        input_image = tf.image.resize(datapoint['image'], (img_size, img_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_size, img_size))

        return normalize(input_image, input_mask)
    
    return dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE),\
            dataset['test'].map(load_image_test),\
            dataset['test'].map(load_image_test)

def return_dataset(datset, img_size):
    return_dict = {"pascal_voc" : import_pascalvoc(img_size),
                   "oxford_pets" : import_oxfordpets(img_size)}
    return return_dict[datset]