import cv2 
import numpy as np 

TEST_IMAGE = cv2.imread('./uploads/board.JPG')
WIDTH = 430
HEIGHT = 335

# Detect the corners of the board and rectify
def corners(image=TEST_IMAGE):
    original = image.copy()

    # Apply yellow mask and convert to binary image
    yellow = yellow_mask(image)
    blur = cv2.bilateralFilter(yellow, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('./uploads/yellow.JPG', yellow)
    # cv2.imwrite('./uploads/gray.JPG', gray)
    # cv2.imwrite('./uploads/binary.JPG', thresh)
    
    # Find all contours in the image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Find the contour of the board - must be a quadrilateral of large enough area
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if area > 150000 and len(approx) == 4:
            cv2.drawContours(image,[c], 0, (36,255,12), 3)
            
            # Extract the four corner points from the approximated polygon
            corners = approx.reshape(4, 2)
            corners = reorder_clockwise(corners)
            
            # Draw a dot on each corner
            for i in range(4):
                x, y = corners[i]
                marked_img = cv2.circle(image, (x,y), radius=35, color=(50, 205, 50), thickness=-1)
            
            # Rectify the image
            corners = np.array(corners, dtype=np.float32)
            dst_points = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners, dst_points)
            rectified_img = cv2.warpPerspective(original, M, (WIDTH, HEIGHT))

            # cv2.imwrite('./uploads/marked.JPG', marked_img)
            # cv2.imwrite('./uploads/rectified.JPG', rectified_img)
 
def yellow_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    YELLOW_LOWER_HSV = np.array([20, 100, 100])
    YELLOW_UPPER_HSV = np.array([30, 255, 255])

    yellow_mask = cv2.inRange(hsv_image, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)
    yellow_masked_img = cv2.bitwise_and(image, image, mask=yellow_mask)
    
    return yellow_masked_img

# Reorder corner points to be clockwise (for cv.getPerspectiveTransform)
def reorder_clockwise(corners):
    # Sort corners based on their y-coordinate
    corners = sorted(corners, key=lambda c: c[1], reverse=True)
    # Divide corners into top and bottom halves
    top_corners = corners[2:]
    bottom_corners = corners[:2]
    # Sort each half based on x-coordinate
    top_corners.sort(key=lambda c: c[0])
    bottom_corners.sort(key=lambda c: c[0])

    sorted_corners = [top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]]
    return sorted_corners