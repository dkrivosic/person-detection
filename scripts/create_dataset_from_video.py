import os

import cv2
import argparse
from PIL import Image
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring
from object_detection.utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
parser.add_argument('--video', dest='video_path', type=str, help='Path to the video file.')
parser.add_argument('--labels', dest='labels_path', type=str,
                                help='Path to the text file containing trajectories.')
parser.add_argument('--output_dir', dest='output_dir', help='Path to the output directory.')
parser.add_argument('--visualize_ground_truth', '-v', dest='visualize_ground_truth', action='store_true',
                                help='Path to the text file containing trajectories.')
args = parser.parse_args()

images_dir = os.path.join(args.output_dir, 'images')
labels_dir = os.path.join(args.output_dir, 'annotations')

if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

cap = cv2.VideoCapture(args.video_path)
_, frame = cap.read()

height, width, _ = frame.shape

horizontal_cells = 10
vertical_cells = 2
cell_width = width // horizontal_cells
cell_height = height // vertical_cells


image_index = 0
labels_by_frame = {}

# Set the frames you want to use in a dataset
frames = list(range(2, 15))

# Read ground truth labels
with open(args.labels_path, 'r') as f:
    for line in f:
        split_line = line.split(' ')
        if(len(split_line) != 11):
            continue

        frame_id = int(split_line[0])
        min_row = int(split_line[2])
        max_row = int(split_line[3])
        min_col = int(split_line[4])
        max_col = int(split_line[5])

        if frame_id not in labels_by_frame:
            labels_by_frame[frame_id] = []

        labels_by_frame[frame_id].append((min_row, max_row, min_col, max_col))

for frame_index in frames:
    _, frame = cap.read()

    x, y = 0, 0
    for x in range(vertical_cells):
        for y in range(horizontal_cells):
            cell = frame[x * cell_height : (x + 1) * cell_height,
                         y * cell_width : (y + 1) * cell_width,
                         :]

            # Skip cells with size 0
            if 0 in cell.shape:
                continue

            image_index += 1
            filename = "%04d.jpg" % (image_index)
            output_path = os.path.join(images_dir, filename)

            # Create XML tree structure
            top = Element('annotation')
            name_tag = SubElement(top, 'filename')
            name_tag.text = filename
            size_tag = SubElement(top, 'size')
            width_tag = SubElement(size_tag, 'width')
            width_tag.text = str(cell_width)
            height_tag = SubElement(size_tag, 'height')
            height_tag.text = str(cell_height)
            depth_tag = SubElement(size_tag, 'depth')
            depth_tag.text = "3"

            boxes = []
            classes = []
            scores = []

            for label in labels_by_frame[frame_index]:
                # Adapt coordinates to subimage
                xmin, xmax, ymin, ymax = label
                xmin -= x * cell_height
                xmax -= x * cell_width
                ymin -= y * cell_width
                ymax -= y * cell_width

                # If only one coordinate is outside of the image, set it to image border
                if xmin < 0 and ymin >= 0 and xmax < cell_height and ymax < cell_width:
                    xmin = 0
                elif xmin >= 0 and ymin < 0 and xmax < cell_height and ymax < cell_width:
                    ymin = 0
                elif xmin >= 0 and ymin >= 0 and xmax >= cell_height and ymax < cell_width:
                    xmax = cell_height - 1
                elif xmin >= 0 and ymin >= 0 and xmax < cell_height and ymax >= cell_width:
                    ymax = cell_width - 1
                elif xmin < 0 or ymin < 0 or xmax >= cell_height or ymax >= cell_width:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(1)
                scores.append(1)

                object_tag = SubElement(top, 'object')
                name_tag = SubElement(object_tag, 'name')
                name_tag.text = 'person'
                bndbox_tag = SubElement(object_tag, 'name')

            # Save XML file
            xml_name = "%04d.xml" % (image_index)
            xml_path = os.path.join(labels_dir, xml_name)
            with open(xml_path, 'w') as f:
                f.write(tostring(top))

            # Visualize ground truth boxes if necessary
            if args.visualize_ground_truth:
                cell = vis_util.visualize_boxes_and_labels_on_image_array(
                    cell,
                    np.array(boxes),
                    classes,
                    None,
                    { 1 : {'name' : 'person'} },
                    instance_masks=None,
                    use_normalized_coordinates=False,
                    line_thickness=1)

            # Save image
            im = Image.fromarray(cell)
            im.save(output_path)

print('Football dataset created.')
