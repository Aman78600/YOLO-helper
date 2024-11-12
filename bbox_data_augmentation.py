import cv2
import time
import os
import sys
from tqdm import tqdm
from glob import glob
import numpy as np
from pathlib import Path
import logging
import getopt

# Setting up logging configuration
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(argv):
    """
    Main function to parse command-line arguments and perform image augmentations.
    """
    help_str = 'yolo_aug.py -i <input_dir> -t <aug_type (90_degree, 180_degree, 270_degree, 90_degree_mirror, 180_degree_mirror, 270_degree_mirror, ori_mirror)> -e <image extension (jpg,jpeg,png ...)> -o <output_dir>'

    try:
        # Parsing command-line arguments
        opts, args = getopt.getopt(
            argv, "hi:t:e:o:", ["input_dir=", "aug_type=", "image_extension=", "output_dir="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)

    input_dir, aug_type, image_ext, output_dir = None, None, None, None

    # Loop through command-line options and assign them to variables
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-i", "--input_dir"):
            input_dir = arg
        elif opt in ("-t", "--aug_type"):
            aug_type = arg
        elif opt in ("-e", "--image_extension"):
            image_ext = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg

    # Check if any required arguments are missing
    if not input_dir or not aug_type or not image_ext or not output_dir:
        log.error('Missing required arguments')
        log.info(help_str)
        sys.exit(2)

    # Check if augmentation type is valid
    if aug_type not in ['90_degree', '180_degree', '270_degree', '90_degree_mirror', '180_degree_mirror', '270_degree_mirror', 'ori_mirror']:
        log.error('Invalid augmentation type')
        log.info(help_str)
        sys.exit(2)

    # Create output directory with a timestamp
    output_dir = os.path.join(
        output_dir, f'{aug_type}_{time.strftime("%H%M%S")}')
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging input and output details
    log.info('Input directory: {}'.format(input_dir))
    log.info('Augmentation type: {}'.format(aug_type))
    log.info('Output directory: {}'.format(output_dir))

    def boxesFromYOLO(imagePath, labelPath):
        """
        Reads an image and its corresponding YOLO format label file, and extracts bounding box coordinates.
        """
        image = cv2.imread(imagePath)  # Read image from the file path
        (hI, wI) = image.shape[:2]  # Get image dimensions
        lines = [line.rstrip('\n') for line in open(
            labelPath)]  # Read label file lines
        boxes = []
        if lines:  # Check if lines are not empty
            for line in lines:
                components = line.split(" ")  # Split line into components
                category = components[0]  # Category of the object
                points = list(map(float, components[1:]))  # Bounding box coordinates
                normalized_points = [(points[i], points[i + 1])
                                     for i in range(0, len(points), 2)]  # Normalize points
                boxes.append((category, normalized_points))  # Append category and points to boxes list
        return (hI, wI, image, boxes)  # Return image dimensions, image object, and boxes

    def rotate_image(image, angle):
        """
        Rotates an image by a specified angle.
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated_image = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_image, new_w, new_h, M

    def mirror_image(image):
        """
        Mirrors an image horizontally.
        """
        mirrored_image = cv2.flip(image, 1)
        return mirrored_image

    def rotate_boxes(boxes, M, W, H, new_W, new_H):
        """
        Rotates bounding boxes by a specified angle using the transformation matrix.
        """
        rotated_boxes = []
        for box in boxes:
            class_name = box[0]
            points = np.array(box[1])

            # Convert normalized coordinates to absolute
            points[:, 0] *= W
            points[:, 1] *= H

            # Apply the rotation matrix
            points = np.hstack([points, np.ones((points.shape[0], 1))])
            new_points = M.dot(points.T).T
            new_points[:, 0] /= new_W
            new_points[:, 1] /= new_H

            rotated_boxes.append((class_name, new_points.tolist()))

        return rotated_boxes

    def mirror_boxes(boxes, W, H):
        """
        Mirrors bounding boxes horizontally.
        """
        mirrored_boxes = []
        for box in boxes:
            class_name = box[0]
            points = [(W - x * W, y * H) for x, y in box[1]]
            points = [(x / W, y / H) for x, y in points]  # Normalize back to [0, 1]
            mirrored_boxes.append((class_name, points))
        return mirrored_boxes

    def convert_boxes_to_yolo_format(boxes):
        yolo_boxes = []
        for box in boxes:
            class_name = box[0]
            coordinates = [str(coord) for point in box[1] for coord in point]
            yolo_boxes.append(f"{class_name} {' '.join(coordinates)}")
        return yolo_boxes

    # Get list of image paths from the input directory
    images = glob(os.path.join(input_dir, f'*.{image_ext}'))

    # Loop through each image and perform augmentation
    for image_path in tqdm(images):
        label_path = image_path.replace(
            f'.{image_ext}', '.txt')  # Get corresponding label file path
        if not os.path.exists(label_path):
            log.warning(
                f'Missing label file for {image_path}')  # Log warning if label file is missing
            continue

        # Read image and bounding boxes from YOLO format
        (hI, wI, image, boxes) = boxesFromYOLO(image_path, label_path)

        # Perform augmentation based on the type
        if aug_type == '90_degree':
            img, new_w, new_h, M = rotate_image(image, 90)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
        elif aug_type == '180_degree':
            img, new_w, new_h, M = rotate_image(image, 180)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
        elif aug_type == '270_degree':
            img, new_w, new_h, M = rotate_image(image, 270)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
        elif aug_type == '90_degree_mirror':
            img, new_w, new_h, M = rotate_image(image, 90)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
            img = mirror_image(img)
            txt_yolo = mirror_boxes(txt_yolo, new_w, new_h)
        elif aug_type == '180_degree_mirror':
            img, new_w, new_h, M = rotate_image(image, 180)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
            img = mirror_image(img)
            txt_yolo = mirror_boxes(txt_yolo, new_w, new_h)
        elif aug_type == '270_degree_mirror':
            img, new_w, new_h, M = rotate_image(image, 270)
            txt_yolo = rotate_boxes(boxes, M, wI, hI, new_w, new_h)
            img = mirror_image(img)
            txt_yolo = mirror_boxes(txt_yolo, new_w, new_h)
        elif aug_type == 'ori_mirror':
            img = image.copy()
            txt_yolo = boxes.copy()
            img = mirror_image(img)
            txt_yolo = mirror_boxes(txt_yolo, wI, hI)

        # Convert boxes to YOLO format
        yolo_boxes = convert_boxes_to_yolo_format(txt_yolo)

        # Generate output file paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(
            output_dir, f'{base_name}_{aug_type}.{image_ext}')
        output_label_path = os.path.join(
            output_dir, f'{base_name}_{aug_type}.txt')

        # Save augmented image
        cv2.imwrite(output_image_path, img)
        # Save augmented bounding boxes in YOLO format
        with open(output_label_path, 'w') as f:
            for line in yolo_boxes:
                f.write(f"{line}\n")


if __name__ == '__main__':
    main(sys.argv[1:])


"""
run command for this script::: 
python yolo_data_preparation.py -i home/...path..../ori_image_with_text_file -t hvflip -e png -o /home/....store_path..../aug_images
"""
