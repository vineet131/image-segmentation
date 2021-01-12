# image-segmentation
Semantic segmentation using CNNs, performed on public datasets like Oxford/IIIT Pets and PASCAL VOC

<h3>About Image Segmentation:</h3>
 
Read about Image Segmentation here (where the modded_unet model was taken from): https://www.tensorflow.org/tutorials/images/segmentation
 
<h3>Dependencies:</h3>

Tensorflow v2.3.1<br>
Tensorboard v2.3<br>
Numpy v1.18.5<br>
Matplotlib v3.3.3<br>
Tensorflow datasets (for Oxford/IIIT Pets dataset) (`pip install tfds-nightly`) v4.1.0<br>

<h3>Steps to download the PASCAL VOC dataset:</h3>

1. Use the shell script at https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/download_and_convert_voc2012.sh or download the datasets directly from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ and build the TFRecords with https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/build_voc2012_data.py

2. Save the TFRecord file at ./datasets/ (Unable to upload the original files due to Github's 25MB file upload limit

<h3>Run the file:</h3>
On command line, set the current directory to where you downloaded this repo and run `python image_segmentation.py`
