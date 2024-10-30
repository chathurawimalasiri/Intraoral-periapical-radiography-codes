class Node:
    # Node class for doubly linked list to store point data
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

def create_doubly_linked_list(points):
    # Creates a circular doubly linked list from a list of points
    if not points:
        return None

    # Initialize the head of the list
    head = Node(points[0])
    current = head

    # Link each point as a node in the list
    for i in range(1, len(points)):
        new_node = Node(points[i])
        current.next = new_node
        new_node.prev = current
        current = new_node

    # Make the list circular
    current.next = head
    head.prev = current

    return head

def adjust_head_to_leftmost(head):
    # Adjusts the head to the leftmost node based on x-coordinates
    if not head:
        return None

    current = head
    leftmost = head
    while True:
        if current.data[0] < leftmost.data[0]:
            leftmost = current
        current = current.next
        if current == head:
            break

    return leftmost

def adjust_head_to_rightmost(head):
    # Adjusts the head to the rightmost node based on x-coordinates
    if not head:
        return None

    current = head
    rightmost = head
    while True:
        if current.data[0] > rightmost.data[0]:
            rightmost = current
        current = current.next
        if current == head:
            break

    return rightmost

import os

# Set directory paths (replace with actual paths)
bone_masks_dir = "PATH_TO_BONE_MASKS_DIRECTORY"   # Directory for bone mask files
bone_lines_output_dir = "PATH_TO_OUTPUT_LINES_DIRECTORY"  # Output for bone line files

# Create output directory if it doesn't exist
os.makedirs(bone_lines_output_dir, exist_ok=True)

# Specify the single file to process (replace with actual filename)
filename = "SINGLE_BONE_MASK_FILENAME.txt"  # The filename of the bone mask file to process
bone_mask_details_path = os.path.join(bone_masks_dir, filename)

converted_bones = []  # To store midpoints for bone lines

# Process bone mask details to generate midpoints
with open(bone_mask_details_path, "r") as f:
    bone_details = f.readlines()
    for line in bone_details:
        # Parse points from each line in the bone mask file
        points = [float(p) for p in line.split(",")]
        points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        n = len(points)

        # Create a circular doubly linked list from the points
        head = create_doubly_linked_list(points)

        # Arrange points from the leftmost and rightmost points
        head = adjust_head_to_leftmost(head)
        
        # Split points into two arrays from the leftmost point
        arr1, arr2 = [], []
        curr1, curr2 = head, head
        for i in range((n+1)//2):
            arr1.append(curr1.data)
            arr2.append(curr2.data)
            curr1 = curr1.next
            curr2 = curr2.prev

        # Now use the rightmost point as the starting point
        head = adjust_head_to_rightmost(head)
        arr3, arr4 = [], []
        curr3, curr4 = head, head
        for i in range((n+1)//2):
            arr3.append(curr3.data)
            arr4.append(curr4.data)
            curr3 = curr3.next
            curr4 = curr4.prev

        # Reverse the second arrays to align correctly for midpoint calculations
        arr3.reverse()
        arr4.reverse()

        # Calculate midpoints for the left and right halves
        mid1 = [((arr1[i][0] + arr2[i][0])/2, (arr1[i][1] + arr2[i][1])/2) for i in range(len(arr1))]
        mid2 = [((arr3[i][0] + arr4[i][0])/2, (arr3[i][1] + arr4[i][1])/2) for i in range(len(arr3))]
        
        # Average the two midpoints to create the final line of points
        mid = [((mid1[i][0] + mid2[i][0])/2, (mid1[i][1] + mid2[i][1])/2) for i in range(len(mid1))]

        # Convert the midpoints to a space-separated string format
        midpoints_str = " ".join([f"{point[0]} {point[1]}" for point in mid])
        converted_bones.append(midpoints_str)

# Save the converted bone lines to a text file
bone_lines_output_path = os.path.join(bone_lines_output_dir, filename)
with open(bone_lines_output_path, "w") as f:
    for bone in converted_bones:
        f.write(bone + "\n")

print("Bone line data saved in:", bone_lines_output_path)
