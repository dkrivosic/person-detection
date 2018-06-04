import os

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from inference import run_inference_for_single_image

horizontal_cells = 10
vertical_cells = 2

flags = tf.app.flags
flags.DEFINE_string('video', None, 'Path to the input video. Leave blank to use camera.')
flags.DEFINE_string('image', None, 'Path to the input image. Leave blank if you want to load a video.')
flags.DEFINE_string('model', '', 'Path to the frozen inference model.')
flags.DEFINE_string('label_map', '', 'Path to the label map.')
FLAGS = flags.FLAGS

# Initialize video loader
is_image = False
if FLAGS.image is not None:
    is_image = True
    frame = cv2.imread(FLAGS.image, cv2.IMREAD_COLOR)
elif FLAGS.video is not None:
    cap = cv2.VideoCapture(FLAGS.video)
else:
    cap = cv2.VideoCapture(0)

is_video = not is_image

if is_video:
    _, frame = cap.read()

height, width, _ = frame.shape
cell_width = width // horizontal_cells
cell_height = height // vertical_cells

# Load Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(FLAGS.model, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print('Model loaded successfully.')

# Load label map
label_map = label_map_util.load_labelmap(FLAGS.label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print('Label map loaded successfully.')

while is_image or cap.isOpened():
    if is_video:
        # Read a video frame
        ret, frame = cap.read()

    # Initialize output frame with zeros
    cells = np.zeros((height, width, 3), dtype='uint8')

    x, y = 0, 0
    for x in range(vertical_cells):
        for y in range(horizontal_cells):

            cell = frame[x * cell_height : (x + 1) * cell_height,
                         y * cell_width : (y + 1) * cell_width,
                         :]

            if 0 in cell.shape:
                continue

            # Run inference on a single cell
            output_dict = run_inference_for_single_image(cell, detection_graph)

            # Keep only detections of people
            ignore = output_dict['detection_classes'] != 1
            output_dict['detection_scores'][ignore] = 0
            output_dict['detection_boxes'][ignore] = np.zeros((4,))
            output_dict['detection_classes'][ignore] = 0


            # Visualization of the results of a detection.
            cell = vis_util.visualize_boxes_and_labels_on_image_array(
                cell,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1)

            # Update cell
            cells[x * cell_height : (x + 1) * cell_height, y * cell_width : (y + 1) * cell_width, :] = cell

    frame = np.array(cells, dtype='uint8')
    cv2.imshow('frame', frame)

    if is_image:
        # cv2.waitKey(0)
        image_name = FLAGS.image.split('.')[0].split('/')[-1] + '_labeled.png'
        cv2.imwrite(image_name, frame)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_video:
    cap.release()
cv2.destroyAllWindows()
