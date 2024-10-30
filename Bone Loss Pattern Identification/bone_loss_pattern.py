import numpy as np
import math
import json
from PIL import Image, ImageDraw, ImageFont
import os

# Constants
LINE_EXTENSION = 20         # number of pixels to extend the tangent line
MASK_SEARCH_DISTANCE = 20   # number of points to search for nearby mask point
SKIP_THRESHOLD = 20         # number of pixels to skip if the closest mask point is too far
ANGULAR_ANGLE = 55          # threshold angle to consider as angular

def classify_image(keypoints):
    maxillary_count = 0
    mandibular_count = 0

    for keypoint_array in keypoints:
        cej_coords = [keypoint_array[0][1], keypoint_array[1][1]]
        apex_coords = []

        if len(keypoint_array) == 6:
            apex_coords = [keypoint_array[4][1], keypoint_array[5][1]]
        elif len(keypoint_array) == 5:
            apex_coords = [keypoint_array[4][1]]

        cej_coords = [x for x in cej_coords if x != 0]
        apex_coords = [x for x in apex_coords if x != 0]

        if len(cej_coords) == 0 or len(apex_coords) == 0:
            return "mandibular"  # assume it's mandibular
        cej_above_apex = sum(cej_y < min(apex_coords) for cej_y in cej_coords)
        cej_below_apex = len(cej_coords) - cej_above_apex

        if cej_above_apex >= cej_below_apex:
            mandibular_count += 1
        else:
            maxillary_count += 1

    if maxillary_count >= mandibular_count:
        return "maxillary"
    else:
        return "mandibular"

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    px, py = polygon[0]
    for i in range(1, n + 1):
        sx, sy = polygon[i % n]
        if min(py, sy) < y <= max(py, sy):
            if x <= max(px, sx):
                if py != sy:
                    xinters = (y - py) * (sx - px) / (sy - py) + px
                if px == sx or x <= xinters:
                    inside = not inside
        px, py = sx, sy
    return inside

def remove_points_inside_masks(bones, masks):
    cleaned_bones = []
    for bone in bones:
        cleaned_bone = []
        for point in bone:
            if not any(point_in_polygon(point, mask) for mask in masks):
                cleaned_bone.append(point)
        if cleaned_bone != []:
            cleaned_bones.append(cleaned_bone)
    return cleaned_bones

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def interpolate(p1, p2, num_points):
    return [
        (
            p1[0] + (p2[0] - p1[0]) * i / (num_points + 1),
            p1[1] + (p2[1] - p1[1]) * i / (num_points + 1)
        )
        for i in range(1, num_points + 1)
    ]

def process_single_image(image_path, bone_line_details_file_path, tooth_mask_details_file_path, keypoints_file_path, output_path, font_path):
    """
    Process a single image with its corresponding bone, mask, and keypoint files
    """
    # Load font
    font = ImageFont.truetype(font_path, 20)
    
    # Get image classification from keypoints
    with open(keypoints_file_path, 'r') as f:
        keypoints_data = json.load(f)
    image_classification = classify_image(keypoints_data["keypoints"])
    
    # Read bone lines
    bones = []
    with open(bone_line_details_file_path, "r") as f:
        bone_details = f.readlines()
        for line in bone_details:
            points = [float(p) for p in line.split()]
            points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
            bones.append(points)

    # Read teeth masks
    masks = []
    with open(tooth_mask_details_file_path, "r") as f:
        mask_details = f.readlines()
        for line in mask_details:
            points = [float(p) for p in line.split()]
            points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

            resampled_points = []
            for i in range(len(points)):
                resampled_points.append(points[i])
                next_point = points[(i + 1) % len(points)]
                
                if distance(points[i], next_point) > 10:
                    num_extra_points = int(distance(points[i], next_point) / 10)
                    resampled_points.extend(interpolate(points[i], next_point, num_extra_points))

            masks.append(resampled_points)

    if len(bones) == 0 or len(masks) == 0:
        raise ValueError("Empty bone or mask details")

    # Load and process the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw bone lines in blue
    for bone in bones:
        draw.line(bone, fill="blue", width=4)

    # Remove points inside masks
    bones = remove_points_inside_masks(bones, masks)

    # Draw masks in green
    for mask in masks:
        draw.polygon(mask, outline="green", width=2)

    # Process each bone
    for bone in bones:
        for end in [0, -1]:  # 0 for start point, -1 for end point
            point = bone[end]

            # Find the closest point on masks
            closest_mask_point = min(
                (p for mask in masks for p in mask),
                key=lambda p: math.dist(point, p)
            )

            if math.dist(point, closest_mask_point) >= SKIP_THRESHOLD:
                continue

            # Find the mask and index of the closest point
            for i, mask in enumerate(masks):
                if closest_mask_point in mask:
                    closest_mask_index = i
                    closest_mask_point_index = mask.index(closest_mask_point)
                    break

            # Get another point on mask
            nearby_mask_point_index = (closest_mask_point_index + MASK_SEARCH_DISTANCE) % len(masks[closest_mask_index])
            nearby_mask_point = masks[closest_mask_index][nearby_mask_point_index]

            # Ensure that nearby_mask_point is in APEX side
            if image_classification == "mandibular":
                if nearby_mask_point[1] < point[1]:
                    nearby_mask_point_index = (closest_mask_point_index - MASK_SEARCH_DISTANCE) % len(masks[closest_mask_index])
                    nearby_mask_point = masks[closest_mask_index][nearby_mask_point_index]
            if image_classification == "maxillary":
                if nearby_mask_point[1] > point[1]:
                    nearby_mask_point_index = (closest_mask_point_index - MASK_SEARCH_DISTANCE) % len(masks[closest_mask_index])
                    nearby_mask_point = masks[closest_mask_index][nearby_mask_point_index]

            # Extend the lines
            start_mask_point = (
                closest_mask_point[0] - LINE_EXTENSION * (closest_mask_point[0] - nearby_mask_point[0]),
                closest_mask_point[1] - LINE_EXTENSION * (closest_mask_point[1] - nearby_mask_point[1])
            )
            end_mask_point = (
                closest_mask_point[0] + LINE_EXTENSION * (closest_mask_point[0] - nearby_mask_point[0]),
                closest_mask_point[1] + LINE_EXTENSION * (closest_mask_point[1] - nearby_mask_point[1])
            )

            # Get mid point index of bone
            mid_point_index = len(bone) // 3
            POINTS_AWAY = mid_point_index

            # Calculate vectors for bone and mask tangent lines
            if end == 0:  # start point
                bone_vector = np.array([bone[POINTS_AWAY][0] - point[0], bone[POINTS_AWAY][1] - point[1]])
            else:  # end point
                bone_vector = np.array([bone[-1-POINTS_AWAY][0] - point[0], bone[-1-POINTS_AWAY][1] - point[1]])
            mask_vector = np.array([end_mask_point[0] - start_mask_point[0], end_mask_point[1] - start_mask_point[1]])

            # Calculate angle
            dot_product = np.dot(bone_vector, mask_vector)
            bone_magnitude = np.linalg.norm(bone_vector)
            mask_magnitude = np.linalg.norm(mask_vector)
            angle_rad = np.arccos(dot_product / (bone_magnitude * mask_magnitude))
            angle_deg = np.degrees(angle_rad)

            # Draw angle information
            angle_text = f"{angle_deg:.2f}°"
            if angle_deg <= ANGULAR_ANGLE:
                circle_center = ((point[0], point[1]))
                circle_radius = 10
                draw.ellipse((
                    circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                    circle_center[0] + circle_radius, circle_center[1] + circle_radius
                ), outline="red", width=3)
                angle_label = "A"
            else:
                angle_label = "H"

            # Calculate text positions
            angle_text_bbox = draw.textbbox((0, 0), angle_text, font=font)
            angle_label_bbox = draw.textbbox((0, 0), angle_label, font=font)
            angle_text_width = angle_text_bbox[2] - angle_text_bbox[0]
            angle_label_width = angle_label_bbox[2] - angle_label_bbox[0]

            # Draw text
            draw.text((point[0], point[1] - 20), angle_text, fill="black", font=font)
            label_position = (point[0] + (angle_text_width - angle_label_width) / 2, point[1] + 5)
            draw.text(label_position, angle_label, fill="black", font=font)

            # Draw lines
            draw.line([start_mask_point, end_mask_point], fill="red", width=2)
            if end == 0:
                draw.line([point, bone[POINTS_AWAY]], fill="yellow", width=2)
            else:
                draw.line([point, bone[-1-POINTS_AWAY]], fill="yellow", width=2)

    # Draw image ID
    image_name = os.path.basename(image_path)
    id_text = f"Image ID: {image_name}"
    id_text_bbox = draw.textbbox((0, 0), id_text, font=font)
    id_text_width = id_text_bbox[2] - id_text_bbox[0]
    id_text_height = id_text_bbox[3] - id_text_bbox[1]
    id_position = (image.width - id_text_width - 10, image.height - id_text_height - 10)
    draw.text(id_position, id_text, fill="black", font=font)

    # Save the modified image
    image.save(output_path)

# Example usage:
if __name__ == "__main__":
    # Set your paths here
    image_path = "path/to/your/image.jpg"
    bone_line_details_file_path = "path/to/your/bone_file.txt"
    tooth_mask_details_file_path = "path/to/your/mask_file.txt"
    keypoints_file_path = "path/to/your/keypoints.json"
    output_path = "path/to/output/image.jpg"
    font_path = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

    try:
        process_single_image(
            image_path,
            bone_line_details_file_path,
            tooth_mask_details_file_path,
            keypoints_file_path,
            output_path,
            font_path
        )
        print(f"Image processed successfully and saved to {output_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
