import os

import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_dir', '', 'Path to the directory containing images')
flags.DEFINE_string('labels_dir', '', 'Path to the directory containing labels')
FLAGS = flags.FLAGS

def create_tf_example_no_annotation(images_dir, filename):
    image_path = os.path.join(images_dir, filename)
    im = Image.open(image_path)
    width, height = im.size
    image_format = b'jpeg' # b'jpeg' or b'png'

    with tf.gfile.GFile(os.path.join(images_dir, filename), 'rb') as f:
        encoded_image_data = f.read() # Encoded image bytes

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature([]),
      'image/object/bbox/xmax': dataset_util.float_list_feature([]),
      'image/object/bbox/ymin': dataset_util.float_list_feature([]),
      'image/object/bbox/ymax': dataset_util.float_list_feature([]),
      'image/object/class/text': dataset_util.bytes_list_feature([]),
      'image/object/class/label': dataset_util.int64_list_feature([]),
      'image/object/difficult': dataset_util.int64_list_feature([])
    }))
    return tf_example

def create_tf_example(example, images_dir):
    height = int(example[5][1].text) # Image height
    width = int(example[5][0].text) # Image width
    filename = example[2].text # Filename of the image. Empty if image is not from file
    image_format = b'jpeg' # b'jpeg' or b'png'

    with tf.gfile.GFile(os.path.join(images_dir, filename), 'rb') as f:
        encoded_image_data = f.read() # Encoded image bytes

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    difficult = []

    objects = example[8:]
    for obj in objects:
        classes_text.append('person'.encode('utf-8'))
        classes.append(1)
        xmin = float(obj[4][0].text) / width
        ymin = float(obj[4][1].text) / height
        xmax = float(obj[4][2].text) / width
        ymax = float(obj[4][3].text) / height
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        difficult.append(0) # The object is not difficult to detect

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult)
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for filename in os.listdir(FLAGS.images_dir):
        if not '.jpg' in filename:
            continue

        xml_path = os.path.join(FLAGS.labels_dir, filename + '.xml')

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            tf_example = create_tf_example(root, FLAGS.images_dir)
        else:
            tf_example = create_tf_example_no_annotation(FLAGS.images_dir, filename)

        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
