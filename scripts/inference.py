import os

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from PIL import Image
from matplotlib import pyplot as plt
from timeit import default_timer as timer

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('model_path', '',
                        'Path to a frozen tensorflow model.')
    flags.DEFINE_string('label_map_path', '',
                        'Path to the label map.')
    flags.DEFINE_string('test_images_dir', '',
                        'Path to the directory with test images.')
    flags.DEFINE_string('output_dir', '',
                        'Path to the output directory.')
    FLAGS = flags.FLAGS

    # Load Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(FLAGS.model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    print('Model loaded successfully.')

    # Load label map
    label_map = label_map_util.load_labelmap(FLAGS.label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('Label map loaded.')


    PATH_TO_TEST_IMAGES = FLAGS.test_images_dir
    TEST_IMAGE_PATHS = sorted(
                        [os.path.join(PATH_TO_TEST_IMAGES, image_name)
                            for image_name in os.listdir(PATH_TO_TEST_IMAGES)
                                if ('.png' in image_name) or ('.jpg' in image_name)]
                        )

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # Inference
    total_images = len(TEST_IMAGE_PATHS)
    image_index = 0
    total_time = 0.0
    for image_path in TEST_IMAGE_PATHS:
        image_index += 1
        print('Running inference for image ' + str(image_index) + '/' + str(total_images))
        image = Image.open(image_path).convert('RGB')
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        start = timer()
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        end = timer()
        total_time += (end - start)

        # Keep only detections of people
        ignore = output_dict['detection_classes'] != 1
        output_dict['detection_scores'][ignore] = 0
        output_dict['detection_boxes'][ignore] = np.zeros((4,))
        output_dict['detection_classes'][ignore] = 0

        # Visualization of the results of a detection.
        image_np = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=1)

        image_name = image_path.split('/')[-1]
        output_path = os.path.join(FLAGS.output_dir, image_name)

        im = Image.fromarray(image_np)
        im.save(output_path)
    print('Average time per image: ' + total_time / total_images)
