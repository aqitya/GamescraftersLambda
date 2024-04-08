# Python program to illustrate 
# corner detection with 
# Harris Corner Detection Method 
  
# organizing imports 
import cv2 
import numpy as np 
  
# path to input image specified and  
# image is loaded with imread command 
testImage = cv2.imread('./uploads/boardPic.png')
  
def detect_corners(image=testImage):
    """"
    # Option 1: Harris Corner Detection
    # source: https://www.geeksforgeeks.org/python-corner-detection-with-harris-corner-detection-method-using-opencv/?ref=lbp
    
    # convert the input image into 
    # grayscale color space 
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # modify the data type 
    # setting to 32-bit floating point 
    operatedImage = np.float32(operatedImage) 
    
    # apply the cv2.cornerHarris method 
    # to detect the corners with appropriate 
    # values as input parameters 
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 
    
    # Results are marked through the dilated corners 
    dest = cv2.dilate(dest, None) 
    
    # Reverting back to the original image, 
    # with optimal threshold value 
    image[dest > 0.01 * dest.max()]=[0, 0, 255] 
    
    # the window showing output image with corners 
    cv2.imshow('Image with Borders', image) 
    
    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows()
    
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to float32 for the algorithm
    gray = np.float32(gray)
    # Detect corners
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst, None)
    # Threshold to get only the significant corners
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    # Display the result
    cv2.imshow('Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
