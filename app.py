import json
import cv2
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # Assuming the video file is stored in /tmp directory within the Lambda environment
    video_path = 'c4cv.mov'
    # Call the modified extract_frames function
    output_strings = extract_frames(video_path)

    # dummy output
    # output_strings = [
    #     "p=1_------X",
    #     "p=2_-O----X",
    #     "p=1_-OX---X",
    #     "p=2_XOX---X",
    # ]
    for string in output_strings:
        print(string)
    
    return {
        "statusCode": 200,
        "body": "\n".join(output_strings)
    }


def grid_to_position_string(grid):
    rows, cols = grid.shape  # Assuming grid is a numpy array with .shape attribute
    position_string = ''

    for col in range(cols):
        for row in range(rows):
            if grid[row][col] == -1:
                position_string += 'X'
            elif grid[row][col] == 1:
                position_string += 'O'
            else:
                position_string += '-'

    return position_string

# Main
def process_frame(frame):
    img = frame

    # Constants
    new_width = 500
    img_h, img_w, _ = img.shape
    scale = new_width / img_w
    img_w = int(img_w * scale)
    img_h = int(img_h * scale)
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
    img_orig = img.copy()

    # Bilateral Filter
    bilateral_filtered_image = cv2.bilateralFilter(img, 15, 190, 190)

    # Calculate the size of each grid cell
    cell_width = img_w // 7
    cell_height = img_h // 6

    # Create a copy of the original image to draw the grid
    grid_image = img_orig.copy()

    # Draw vertical lines for the grid
    for i in range(1, 7):
        cv2.line(grid_image, (i * cell_width, 0), (i * cell_width, img_h), (0, 255, 0), 1)

    # Draw horizontal lines for the grid
    for i in range(1, 6):
        cv2.line(grid_image, (0, i * cell_height), (img_w, i * cell_height), (0, 255, 0), 1)

    # Display the image with the grid
    cv2.imwrite('grid_image_with_cells.png', grid_image)


    # Initialize the grid
    grid = np.zeros((6, 7))

    BLACK_LOWER_HSV = np.array([100, 150, 50])
    BLACK_UPPER_HSV = np.array([140, 255, 255])

    RED_LOWER_HSV = np.array([0, 120, 70])
    RED_UPPER_HSV = np.array([10, 255, 255])

    # Function to check if the majority of the pixels in a masked area are of the color of the mask
    def is_color_dominant(mask):
        # Count the non-zero (white) pixels in the mask
        white_pixels = cv2.countNonZero(mask)
        # Calculate the percentage of white pixels
        white_area_ratio = white_pixels / mask.size
        # If the white area covers more than 20% of the mask, we consider the color to be dominant
        return white_area_ratio > 0.2

    def process_cell(img, x_start, y_start, width, height):
        # Crop the cell from the image
        cell_img = img[y_start:y_start + height, x_start:x_start + width]

        # Convert the cell image to grayscale
        gray_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # Apply Hough Circle Transform to find circles in the grayscale image
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                param1=50, param2=30, minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Ensure the crop coordinates are within the image bounds
                x, y, r = int(x), int(y), int(r)
                x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(width, x + r), min(height, y + r)

                # Crop the circle from the cell_img
                circle_img = cell_img[y1:y2, x1:x2]

                # If the circle_img is empty, skip to the next circle
                if circle_img.size == 0:
                    continue

                # Convert to HSV and create masks for red and black
                hsv_circle_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2HSV)
                red_mask = cv2.inRange(hsv_circle_img, RED_LOWER_HSV, RED_UPPER_HSV)
                black_mask = cv2.inRange(hsv_circle_img, BLACK_LOWER_HSV, BLACK_UPPER_HSV)

                if is_color_dominant(red_mask):
                    return 1  # Red 
                elif is_color_dominant(black_mask):
                    return -1  # Black 

        # No circles detected or not the right color
        return 0

    # Analyze each cell and update the grid
    for row in range(6):
        for col in range(7):
            x_start = col * cell_width
            y_start = row * cell_height
            grid[row, col] = process_cell(bilateral_filtered_image, x_start, y_start, cell_width, cell_height)
    
    return grid

# determines if there is motion in the frame
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Could be better...
def is_motion(previous_frame, current_frame, threshold=10):
    frame_delta = cv2.absdiff(previous_frame, current_frame)
    thresholded = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_level = np.sum(thresholded)
    return motion_level > threshold


# Need to compelte find_board_bounds
def find_board_bounds(frame, threshold=50):
    x, y = 0, 0  # Top left corner
    w, h = 430, 335  # Width and height calculated from bottom right - top left
    
    return x, y, w, h


def check_winner(grid):
    rows, cols = len(grid), len(grid[0])

    grid_array = np.array(grid)

    for row in grid_array:
        for i in range(cols - 3):
            if np.array_equal(row[i:i + 4], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(row[i:i + 4], np.array([-1, -1, -1, -1])):
                return -1

    for col in range(cols):
        for i in range(rows - 3):
            if np.array_equal(grid_array[i:i + 4, col], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal(grid_array[i:i + 4, col], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(rows - 3):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row + i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1

    for row in range(3, rows):
        for col in range(cols - 3):
            if np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([1, 1, 1, 1])):
                return 1
            elif np.array_equal([grid_array[row - i][col + i] for i in range(4)], np.array([-1, -1, -1, -1])):
                return -1
    return 0


def extract_frames(video_path, skip_frames=20):
    cap = cv2.VideoCapture(video_path)
    previous_position_string = None
    output_strings = []  # List to store output strings

    ret, previous_frame = cap.read()
    if not ret:
        output_strings.append("Error: Cannot read frame from video.")
        cap.release()
        return output_strings
    
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    previous_frame = cv2.GaussianBlur(previous_frame, (21, 21), 0)
    previous_position_string = '------------------------------------------'
    p, frame_count = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if frame_count % skip_frames == 0:
            if not is_motion(previous_frame, gray):
                board_array = process_frame(frame)
                current_position_string = grid_to_position_string(board_array)
                
                if current_position_string != previous_position_string:
                    output_strings.append(f'p={str((p % 2) + 1)}_{current_position_string}')
                    p += 1

                result = check_winner(board_array)
                if result == -1:
                    output_strings.append("Player 2 RED wins!")
                    break
                elif result == 1:
                    output_strings.append("Player 1 BLUE wins!")
                    break

                previous_position_string = current_position_string
            previous_frame = gray
        frame_count += 1

    cap.release()
    return output_strings
