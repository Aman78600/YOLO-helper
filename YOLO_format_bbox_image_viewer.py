import cv2
import os

# Function to convert YOLO format bbox (normalized) to pixel coordinates
def yolo_to_bbox(bbox, img_w, img_h):
    x_center, y_center, width, height = bbox
    x_center = x_center * img_w
    y_center = y_center * img_h
    width = width * img_w
    height = height * img_h

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return x_min, y_min, x_max, y_max

# Function to visualize bounding boxes
def visualize_bounding_boxes(image_path, txt_path):
    # Load the image
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    # Read the bounding box annotations from .txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Loop through all bounding boxes
    for line in lines:
        bbox = line.strip().split()

        try:
            class_id = int(float(bbox[0]))  # Class ID should be an integer, but some files might have it as float
        except ValueError:
            print(f"Skipping invalid line in {txt_path}: {line}")
            continue

        bbox_coords = list(map(float, bbox[1:]))  # YOLO bbox: x_center, y_center, width, height
        
        # Convert YOLO bbox format to (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = yolo_to_bbox(bbox_coords, img_w, img_h)

        # Draw the bounding box on the image (BGR format for OpenCV)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

        # Optionally, add the class label above the bounding box
        label = f"Class: {class_id}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Example usage
if __name__ == "__main__":
    image_path = ''  # Path to your image
    txt_path = ''    # Path to your corresponding .txt file

    visualize_bounding_boxes(image_path, txt_path)
