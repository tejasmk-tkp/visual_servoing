import pyzed.sl as sl
import cv2
import numpy as np
import signal
import sys

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global running
    print("\nShutdown signal received. Cleaning up...")
    running = False

def draw_detection_info(image, obj, image_scale=1.0):
    """
    Draw bounding boxes and object information on the image
    """
    # Colors for different detection states
    colors = {
        'OK': (0, 255, 0),      # Green
        'SEARCHING': (0, 255, 255),  # Yellow
        'OFF': (128, 128, 128)   # Gray
    }
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    
    try:
        # Get the 3D bounding box and convert to 2D for display
        bbox_3d = np.array([[float(coord) for coord in pt] for pt in obj.bounding_box])
        bbox_2d = bbox_3d[:, :2]  # Use only x,y coordinates
        
        # Scale points to image coordinates
        bbox_2d = (bbox_2d * image_scale).astype(np.int32)
        
        # Get color based on tracking state
        color = colors.get(str(obj.tracking_state), colors['OFF'])
        
        # Draw the bounding box
        # Draw bottom rectangle
        cv2.polylines(image, [bbox_2d[:4]], True, color, 2)
        # Draw top rectangle
        cv2.polylines(image, [bbox_2d[4:]], True, color, 2)
        # Draw vertical lines
        for i in range(4):
            cv2.line(image, tuple(bbox_2d[i]), tuple(bbox_2d[i+4]), color, 2)
        
        # Print detection details to console
        print(f"\nDetection Details:")
        print(f"ID: {obj.id}")
        print(f"Label: {obj.label}")
        print(f"Confidence: {obj.confidence:.2f}%")
        print(f"Tracking State: {obj.tracking_state}")
        print(f"Position (xyz): {obj.position}")
        print(f"Velocity (xyz): {obj.velocity}")
        print(f"Bounding Box 3D: {bbox_3d}")
        print("-" * 50)
        
        # Prepare text information
        info_text = [
            f"ID: {obj.id}",
            f"Label: {obj.label}",
            f"Conf: {int(obj.confidence)}%",
            f"Vel: {obj.velocity[0]:.2f}, {obj.velocity[1]:.2f}, {obj.velocity[2]:.2f}"
        ]
        
        # Draw text
        text_x = bbox_2d[0][0]
        text_y = bbox_2d[0][1] - 10
        for i, text in enumerate(info_text):
            y = text_y - (i * 20)
            cv2.putText(image, text, (text_x, y), font, font_scale, color, font_thickness)
            
    except Exception as e:
        print(f"Error drawing detection info: {e}")

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
                        # Draw detection information on image
                        draw_detection_info(cv_image, obj)
                            
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
