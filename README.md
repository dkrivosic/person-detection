# Person detection using Tensorflow object detection API

## Usage

1. Create tensorflow record files from your images and labels. generate_records.py works with label files that are formatted like Pascal VOC labels.
```
python scripts/generate_records.py \
    --images_dir ../football_data/images \
    --labels_dir ../football_data/labels \
    --output_path data/train.record
```

2. Train a model.
```
python ~/models/research/object_detection/train.py \
    --logtostderr --pipeline_config_path models/model/faster_rcnn_inception_v2_coco.config \
    --train_dir models/train
```

3. Export inference graph.
```
python ~/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/model/faster_rcnn_inception_v2_coco.config \
    --trained_checkpoint_prefix models/train/model.ckpt-10 \
    --output_directory models/eval/my_model
```

4. Run inference script.
```
python scripts/inference.py \
    --model_path models/eval/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb \
    --label_map_path data/person_label_map.pbtxt \
    --test_images_dir ../football_data/images \
    --output_dir out_images
```
