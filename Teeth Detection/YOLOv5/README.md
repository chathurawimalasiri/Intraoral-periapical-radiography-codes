# Tooth Detection Dataset for YOLOv8

This dataset is used to train a YOLOv5 model to detect teeth in dental images. The dataset contains images of teeth along with their corresponding annotations. The code can be found at [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

## Dataset Structure

```plaintext
 data/
  ├── test/
  │   ├── images/
  │   │   └── *.jpg/png
  │   └── labels/
  │       └── *.txt
  ├── train/
  │   ├── images/
  │   │   └── *.jpg/png
  │   └── labels/
  │       └── *.txt
  └── val/
      ├── images/
      │   └── *.jpg/png
      └── labels/
          └── *.txt
```

## Annotation Format

Each image has a corresponding .txt file in the labels folder with the same name.
Example: if image is "image001.jpg", annotation is "image001.txt"

Here's how the annotation format should look for each tooth instance in `labels/*.txt`, along with the example and normalization details:

---

### Format for Each Tooth Instance Annotation File in `labels/*.txt`

Each line in a `.txt` file represents a tooth instance segmentation, with the following format:

```plaintext
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- **class_id**: Class identifier. Since each instance is a tooth, use `0`.
- **xi, yi**: Normalized segmentation coordinates

### Example Annotation File (labels/image001.txt)

Here is an example with a single tooth instance:

```plaintext
0 0.568 0.123 0.568 0.456 0.789 0.456 0.789 0.123 ...
```

### Coordinate Normalization

To obtain the normalized values for any point `(x_pixel, y_pixel)`:

- **Normalized x-coordinate** = `x_pixel / image_width`
- **Normalized y-coordinate** = `y_pixel / image_height`
