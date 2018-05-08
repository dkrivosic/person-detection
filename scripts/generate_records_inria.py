import tensorflow as tf
from object_detection.utils import dataset_util
import os
import re

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('root_dir', '', 'Path to the INRIA root directory')
flags.DEFINE_string('labels_dir', '', 'Path to the directory containing labels')
FLAGS = flags.FLAGS

def create_tf_example(lines, root_dir):
    filename = re.search(r'[a-zA-Z0-9-_\/]+\.[a-zA-Z]+', lines[2].strip()).group(0)
    dimensions = lines[3].strip().split(':')[1].split('x') # Filename of the image.
    width = int(dimensions[0].strip()) # Image width
    height = int(dimensions[1].strip()) # Image height
    image_format = b'png' # b'jpeg' or b'png'

    with tf.gfile.GFile(os.path.join(root_dir, filename), 'rb') as f:
        encoded_image_data = f.read() # Encoded image bytes

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    difficult = []

    object_index = 17 # Groundtruth box for object 1 is at line 17

    while object_index < len(lines):
        bounding_box = lines[object_index].strip().split(':')[1]
        bounding_box = bounding_box.replace('(', '').replace(')', '')
        minimums = bounding_box.split('-')[0]
        maximums = bounding_box.split('-')[1]
        xmin = int(minimums.split(',')[0].strip())
        xmin = float(xmin) / width
        xmins.append(xmin)
        ymin = int(minimums.split(',')[1].strip())
        ymin = float(ymin) / height
        ymins.append(ymin)
        xmax = int(maximums.split(',')[0].strip())
        xmax = float(xmax) / width
        xmaxs.append(xmax)
        ymax = int(maximums.split(',')[1].strip())
        ymax = float(ymax) / height
        ymaxs.append(ymax)
        object_index += 7
        classes.append(1)
        classes_text.append('person'.encode('utf-8'))
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

    for filename in os.listdir(FLAGS.labels_dir):
        with open(os.path.join(FLAGS.labels_dir, filename), 'r') as f:
            lines = f.readlines()

        tf_example = create_tf_example(lines, FLAGS.root_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
