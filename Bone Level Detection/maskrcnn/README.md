# Bone level Detection with MaskRCNN

To detect bone levels, an instance segmentation model is trained using the Mask R-CNN architecture. The code is implemented in PyTorch and can be found at [PyTorch-Simple-MaskRCNN](https://github.com/Piumal1999/PyTorch-Simple-MaskRCNN).

## Dataset Structure

```plaintext
data/
├── annotations/
│   ├── instances_test.json
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg 
└── test/
    └── *.jpg
```

## Annotation Format

The dataset and the annotations are provided in COCO dataset format.

The COCO JSON file has the following main sections:

1. **Info**: General dataset information.

2. **Licenses**: Information about image licenses (optional).

3. **Images**: Each entry describes an image in the dataset.
   - Key fields:
     - `id`: Unique identifier for each image.
     - `width` and `height`: Dimensions of the image.
     - `file_name`: Image file name (e.g., `"image_001.jpg"`).
   - Example:

     ```json
     "images": [
         {
             "id": 1,
             "width": 1024,
             "height": 768,
             "file_name": "image_001.jpg"
         }
     ]
     ```

4. **Annotations**: Contains the details of each annotation (object instance) in the images.
   - Key fields:
     - `id`: Unique identifier for each annotation.
     - `image_id`: ID of the image where the object appears.
     - `category_id`: ID of the category the object belongs to.
     - `segmentation`: List of pixel coordinates that define the object's segmentation mask, formatted as a list of lists.
     - `bbox`: Bounding box around the object, specified as `[x, y, width, height]`.
     - `area`: Area of the object, in pixels.
     - `iscrowd`: Indicates if the object is a crowd segment (0 for single instances, 1 for a crowd).
   - Example:

     ```json
     "annotations": [
         {
             "id": 1,
             "image_id": 1,
             "category_id": 1,
             "segmentation": [[120.0, 200.0, 130.0, 210.0, ...]],
             "area": 500,
             "bbox": [100.0, 150.0, 50.0, 60.0],
             "iscrowd": 0
         }
     ]
     ```

5. **Categories**: Defines the categories (classes) in the dataset.
   - Key fields:
     - `id`: Unique identifier for each category.
     - `name`: Name of the category (e.g., `"bone"`).
     - `supercategory`: Higher-level grouping for categories (optional).
   - Example:

     ```json
     "categories": [
         {
             "id": 1,
             "name": "bone",
             "supercategory": "dental"
         }
     ]
     ```

### COCO JSON Example Overview

A simplified example the annotation file might look like this:

```json
{
    "info": { ... },
    "licenses": [ ... ],
    "images": [
        {
            "id": 1,
            "width": 1024,
            "height": 768,
            "file_name": "image_001.jpg"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [[120.0, 200.0, 130.0, 210.0, ...]],
            "area": 500,
            "bbox": [100.0, 150.0, 50.0, 60.0],
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "bone",
            "supercategory": "dental"
        }
    ]
}
```
