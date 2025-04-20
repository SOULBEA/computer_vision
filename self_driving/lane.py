import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 120)
    return canny

def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2= line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    
    # Define polygon as a numpy array with explicit int32 data type
    # Make sure it's a 3D array: [number of polygons][number of points][x,y]
    polygons = np.array([
        [[200, height], [2000, height], [550, 250]]
    ], dtype=np.int32)
    
    # Create a mask with same dimensions as input image
    mask = np.zeros_like(image)
    
    # Fill the polygon with white (255)
    # For grayscale, we need a single channel mask
    if len(image.shape) > 2:
        # Color image
        cv2.fillPoly(mask, polygons, (255, 255, 255))
    else:
        # Grayscale image
        cv2.fillPoly(mask, polygons, 255)
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Main execution
try:
    image = cv2.imread("lane1.png")
    if image is None:
        raise Exception("Could not read image file")
    
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_line(lane_image, lines)
    
    # Test the region_of_interest function directly
    cv2.imshow("result", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
except Exception as e:
    print(f"Error: {e}")
