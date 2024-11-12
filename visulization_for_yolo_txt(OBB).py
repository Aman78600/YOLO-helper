import cv2
import os
import numpy as np

# Function to visualize oriented bounding boxes
def visualize_obb(image_path, txt_path):
    # Load the image
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    # Read the bounding box annotations from the .txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Loop through all bounding boxes
    for line in lines:
        bbox = line.strip().split()

        try:
            class_id = int(float(bbox[0]))  # Class ID
        except ValueError:
            print(f"Skipping invalid line in {txt_path}: {line}")
            continue

        # Oriented Bounding Box: [x1, y1, x2, y2, x3, y3, x4, y4]
        obb_coords = list(map(float, bbox[1:]))  # Convert all coordinates to float
        obb_coords = np.array(obb_coords).reshape(-1, 2)  # Reshape to a 4x2 array

        # # Convert normalized coordinates to pixel coordinates
        obb_coords[:, 0] *= img_w  # Scale x-coordinates by image width
        obb_coords[:, 1] *= img_h  # Scale y-coordinates by image height


        
        # Draw the polygon for the OBB (4 corners)
        obb_coords = obb_coords.astype(int)  # Convert to integer pixel values
        cv2.polylines(image, [obb_coords], isClosed=True, color=(0, 255, 0), thickness=2)  # Draw green OBB
    
        # Optionally, add the class label near the first point of the bounding box
        label = f"Class: {class_id}"
        cv2.putText(image, label, (obb_coords[0][0], obb_coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # break

    # Show the image with oriented bounding boxes
    # cv2.imshow('Image with Oriented Bounding Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
   
    try:
        os.mkdir('visulisation_image')
    except:
        pass
    cv2.imwrite(f'visulisation_image/{image_path.split("/")[-1]}', image)
    
# Example usage
if __name__ == "__main__":

    image_path = '.png'  # Path to your image
    txt_path = '.txt'    # Path to your corresponding .txt file
    visualize_obb(image_path,txt_path)
