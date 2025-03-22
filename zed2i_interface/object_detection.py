import pyzed.sl as sl
import cv2
import numpy as np
import signal
import sys
from datetime import datetime

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global running
    print("\nShutdown signal received. Cleaning up...")
    running = False

def log_object_data(obj):
    """
    Log detailed information about detected objects
    """
    try:
        # Create a timestamp for the log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get position and velocity
        position = obj.position
        velocity = obj.velocity
        
        # Basic object information
        print("\n" + "="*50)
        print(f"Timestamp: {timestamp}")
        print(f"Object ID: {obj.id}")
        print(f"Label: {obj.label}")
        print(f"Sublabel: {obj.sublabel}")
        print(f"Tracking State: {obj.tracking_state}")
        print(f"Confidence: {obj.confidence:.2f}%")
        
        # Position information
        print("\nPosition (meters):")
        print(f"  X: {position[0]:.3f}")
        print(f"  Y: {position[1]:.3f}")
        print(f"  Z: {position[2]:.3f}")
        
        # Velocity information
        print("\nVelocity (m/s):")
        print(f"  X: {velocity[0]:.3f}")
        print(f"  Y: {velocity[1]:.3f}")
        print(f"  Z: {velocity[2]:.3f}")
        
        # Dimensions
        dimensions = obj.dimensions
        print("\nDimensions (meters):")
        print(f"  Length: {dimensions[0]:.3f}")
        print(f"  Height: {dimensions[1]:.3f}")
        print(f"  Width: {dimensions[2]:.3f}")
        
        '''# Bounding box information
        bbox = obj.bounding_box
        print("\nBounding Box Corners (3D, meters):")
        for i, point in enumerate(bbox):
            print(f"  Corner {i+1}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")'''
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error logging object data: {e}")

def draw_3d_bounding_box(image, bbox_2d, color):
    """
    Draw a 3D bounding box as a cuboid
    """
    try:
        # Draw bottom rectangle (points 0-3)
        for i in range(4):
            pt1 = tuple(map(int, bbox_2d[i]))
            pt2 = tuple(map(int, bbox_2d[(i + 1) % 4]))
            cv2.line(image, pt1, pt2, color, 2)
        
        # Draw top rectangle (points 4-7)
        for i in range(4):
            pt1 = tuple(map(int, bbox_2d[i + 4]))
            pt2 = tuple(map(int, bbox_2d[((i + 1) % 4) + 4]))
            cv2.line(image, pt1, pt2, color, 2)
        
        # Draw vertical lines connecting top and bottom rectangles
        for i in range(4):
            pt1 = tuple(map(int, bbox_2d[i]))
            pt2 = tuple(map(int, bbox_2d[i + 4]))
            cv2.line(image, pt1, pt2, color, 2)

    except Exception as e:
        print(f"Error in draw_3d_bounding_box: {e}")

def draw_detection_info(image, obj, camera):
    """
    Draw bounding boxes and object information on the image using proper 3D-2D projection
    """
    try:
        # Colors for different detection states
        colors = {
            sl.OBJECT_TRACKING_STATE.OK: (0, 255, 0),        # Green
            sl.OBJECT_TRACKING_STATE.SEARCHING: (0, 255, 255),  # Yellow
            sl.OBJECT_TRACKING_STATE.OFF: (128, 128, 128)    # Gray
        }
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        # Get color based on tracking state
        color = colors.get(obj.tracking_state, colors[sl.OBJECT_TRACKING_STATE.OFF])

        # Get image size
        height, width = image.shape[:2]

        # Get 3D bounding box points and convert to image space
        bbox_3d = obj.bounding_box
        bbox_2d = []
        
        for point3D in bbox_3d:
            # Scale and transform the points to image space
            x, y, z = point3D
            if z != 0:
                x = ((x / z) * 1000) + (width / 2)   # Scale factor of 1000 and center offset
                y = ((y / z) * 1000) + (height / 2)  # Adjust these values if needed
                bbox_2d.append([x, y])
            else:
                bbox_2d.append([0, 0])  # Fallback for zero depth

        # Draw the 3D bounding box if we have valid points
        if len(bbox_2d) == 8:  # Make sure we have all 8 points
            draw_3d_bounding_box(image, bbox_2d, color)
        
            # Draw text information
            text_x = int(bbox_2d[0][0])
            text_y = int(bbox_2d[0][1]) - 10
            info_text = [
                f"ID: {obj.id}",
                f"Label: {obj.label}",
                f"Conf: {int(obj.confidence)}%"
            ]
            
            for i, text in enumerate(info_text):
                y = text_y - (i * 20)
                cv2.putText(image, text, (text_x, y), font, font_scale, color, font_thickness)
            
    except Exception as e:
        print(f"Error drawing detection info: {e}")
        import traceback
        traceback.print_exc()

def main():
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and configure Camera object
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {repr(err)}. Exit program.")
        return

    try:
        # Configure object detection
        obj_param = sl.ObjectDetectionParameters()
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

        if obj_param.enable_tracking:
            positional_tracking_param = sl.PositionalTrackingParameters()
            zed.enable_positional_tracking(positional_tracking_param)

        print("Object Detection: Loading Module...")
        err = zed.enable_object_detection(obj_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Enable object detection failed: {repr(err)}. Exit program.")
            return

        # Create objects for detection and image retrieval
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 40
            
        # Image objects
        image = sl.Mat()
            
        # Create window for display
        cv2.namedWindow("ZED Object Detection", cv2.WINDOW_NORMAL)

        print("\nStarting detection loop...")
        print("Press ESC or Ctrl+C to exit")
        
        while running:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Retrieve image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                cv_image = image.get_data()
                    
                # Retrieve objects
                zed.retrieve_objects(objects, obj_runtime_param)
                    
                if objects.is_new:
                    print(f"\nDetected {len(objects.object_list)} objects")
                    # Process each detected object
                    for obj in objects.object_list:
                        # Log detailed object information
                        log_object_data(obj)
                        # Draw detection information on image
                        draw_detection_info(cv_image, obj, zed)
                            
                # Display the image
                cv2.imshow("ZED Object Detection", cv_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("\nESC pressed. Exiting...")
                    running = False
                    break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        
    finally:
        # Clean up
        print("\nCleaning up...")
        zed.disable_object_detection()
        zed.close()
        cv2.destroyAllWindows()
        print("Program finished successfully")
                        
if __name__ == "__main__":
    main()
