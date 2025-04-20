import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    
    # Define a triangular region of interest
    # You may need to adjust these values based on your images
    polygons = np.array([
        [(0, height), (width, height), (width // 2, height // 2)]
    ])
    
    # Create a mask with the same dimensions as our image
    mask = np.zeros_like(image)
    
    # Fill the polygon white
    cv2.fillPoly(mask, polygons, 255)
    
    # Apply the mask to our image
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def detect_line_segments(image):
    # Define parameters for Hough Line transformation
    rho = 2  # distance resolution in pixels
    theta = np.pi / 180  # angular resolution in radians
    threshold = 100  # minimum number of votes to be considered a line
    min_line_length = 40  # minimum line length
    max_line_gap = 5  # maximum gap between line segments
    
    # Apply Hough Line transform
    line_segments = cv2.HoughLinesP(image, rho, theta, threshold, 
                                  np.array([]), min_line_length, max_line_gap)
    
    return line_segments

def average_slope_intercept(line_segments, image):
    lane_lines = []
    
    if line_segments is None:
        return lane_lines
    
    height, width = image.shape[:2]
    left_fit = []
    right_fit = []
    
    # Define the boundary to separate left and right lanes
    boundary = width / 2
    
    # Left lane: slope is negative
    # Right lane: slope is positive
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue  # Skip vertical lines
            
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            if slope < 0:
                if x1 < boundary and x2 < boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > boundary and x2 > boundary:
                    right_fit.append((slope, intercept))
    
    # Calculate average slope and intercept for left and right lane
    left_fit_average = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_average = np.average(right_fit, axis=0) if len(right_fit) > 0 else None
    
    # Create lines that extend from the bottom of the image to the horizon
    if left_fit_average is not None:
        slope, intercept = left_fit_average
        y1 = height  # Bottom of the image
        y2 = int(height * 0.6)  # Slightly above the middle
        x1 = max(0, min(width, int((y1 - intercept) / slope)))
        x2 = max(0, min(width, int((y2 - intercept) / slope)))
        lane_lines.append([[x1, y1, x2, y2]])
    
    if right_fit_average is not None:
        slope, intercept = right_fit_average
        y1 = height  # Bottom of the image
        y2 = int(height * 0.6)  # Slightly above the middle
        x1 = max(0, min(width, int((y1 - intercept) / slope)))
        x2 = max(0, min(width, int((y2 - intercept) / slope)))
        lane_lines.append([[x1, y1, x2, y2]])
    
    return lane_lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    return line_image

def lane_detection_pipeline(image):
    # Step 1: Convert to Canny edges
    canny_image = canny_edge_detector(image)
    
    # Step 2: Extract region of interest
    cropped_image = region_of_interest(canny_image)
    
    # Step 3: Detect line segments
    line_segments = detect_line_segments(cropped_image)
    
    # Step 4: Average and extrapolate lines
    lane_lines = average_slope_intercept(line_segments, image)
    
    # Step 5: Visualize the results
    line_image = display_lines(image, lane_lines)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    return result

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    result = lane_detection_pipeline(image_rgb)
    
    # Display the results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title('Lane Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

def process_video(video_path, output_path):
    cap = cv2.VideoCapture("lane.mp4")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = lane_detection_pipeline(frame_rgb)
        
        # Convert back to BGR for saving
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the output video
        out.write(processed_frame_bgr)
        
        # Display the processed frame
        cv2.imshow('Lane Detection', processed_frame_bgr)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # For processing a single image
    # Replace 'road_image.jpg' with your image path
    # process_image('road_image.jpg')
    
    # For processing a video
    # Replace 'road_video.mp4' with your video path
    process_video('lane.mp4', 'output_video.avi')
    
    # For testing with a webcam
    def test_webcam():
        cap = cv2.VideoCapture(0)  # 0 for default webcam
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = lane_detection_pipeline(frame_rgb)
            
            # Convert back to BGR for display
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            # Display the processed frame
            cv2.imshow('Lane Detection', processed_frame_bgr)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    # Uncomment to test with webcam
    # test_webcam()
    
    print("Please uncomment one of the example usages to process an image, video, or webcam feed.")
