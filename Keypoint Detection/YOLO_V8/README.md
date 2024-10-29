# Tooth Detection Dataset for YOLOv8

## Dataset Structure

```
data/
├── images/
│   ├── train/
│   │   └── *.jpg/png  
│   ├── val/
│   │   └── *.jpg/png
│   └── test/
│       └── *.jpg/png
└── labels/
    ├── train/
    │   └── *.txt  
    ├── val/
    │   └── *.txt
    └── test/
        └── *.txt
```

## Annotation Format

Each image has a corresponding .txt file in the labels folder with the same name.
Example: if image is "image001.jpg", annotation is "image001.txt"

### Format for each tooth instance in labels/*.txt:

`<class>` <cej1_x> <cej1_y> <cej1_v> <cej2_x> <cej2_y> <cej2_v> <apex1_x> <apex1_y> <apex1_v> <apex2_x> <apex2_y> <apex2_v>

where:

- class: tooth instance index (0 for tooth instance)
- Four keypoints per tooth instance in this order:
  1. CEJ 1 (x, y, visibility)
  2. CEJ  2(x, y, visibility)
  3. Apex 1(x, y, visibility)
  4. Apex  2(x, y, visibility)
- All x, y coordinates are normalized (0.0-1.0)
- visibility flag values:
  0 = not visible
  1 = visible but not labeled
  2 = visible and labeled

### Example annotation file (labels/image001.txt):

```
# Single tooth instance with 4 keypoints 
0 0.425 0.532 2 0.428 0.535 2 0.426 0.678 2 0.429 0.682 2
```

### Coordinate Normalization:

- x = x_pixel / image_width
- y = y_pixel / image_height

