import pyzed.sl as sl
import cv2
import numpy as np
import signal
import sys
from datetime import datetime
from ultralytics import YOLO

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global running
    print("\nShutdown signal received. Cleaning up...")
    running = False

def log_object_data(obj, class_name="Unknown"):
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
        print(f"Object Class Name: {class_name}")
        print(f"Label: {obj.raw_label}")
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

def draw_detection_info(image, obj, camera, class_name="Unknown"):
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
                f"Label: {class_name}",
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

    model = YOLO("yolov8m.pt")

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

    # Create image and depth matrices
    width = 1280
    height = 720
    
    image_zed = sl.Mat(width, height, sl.MAT_TYPE.U8_C4)
    depth_zed = sl.Mat(width, height, sl.MAT_TYPE.F32_C1)

    # Ensure Object Detection is enabled in ZED (only do this once)
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True  # 🔥 Required for `instance_id`
    zed.enable_object_detection(obj_param)
    
    # Create display window
    cv2.namedWindow("ZED YOLO Object Detection", cv2.WINDOW_NORMAL)
    
    print("\nStarting YOLO detection and ingestion loop...")
    print("Press ESC or Ctrl+C to exit")
    
    while running:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image and depth map
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            cv_image = image_zed.get_data()
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            depth_map = depth_zed.get_data()

            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            
            # Run ultralytics YOLO on the image
            results = model(cv_image_rgb, verbose=False)
            # Assuming the first result is what we want
            detections = results[0].boxes  # YOLOBoxes object
            # The YOLOBoxes object contains xyxy, conf, and cls for each detection
            boxes = []
            confidences = []
            class_ids = []
            
            for box in detections.xyxy.cpu().numpy():
                # box format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.astype(int)
                w = x2 - x1
                h = y2 - y1
                boxes.append([x1, y1, w, h])
            for conf in detections.conf.cpu().numpy():
                confidences.append(float(conf))
            for cls in detections.cls.cpu().numpy():
                class_ids.append(int(cls))

            # Track instances of detected objects
            tracked_objects = {}  # Dictionary to store previous object IDs
            
            # Create custom box objects for ingestion into ZED
            objects_in = []

            for i, box in enumerate(detections.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                conf = float(detections.conf[i].cpu().numpy())
                cls_id = int(detections.cls[i].cpu().numpy())
                class_name = model.names[cls_id]  # YOLO class name
                
                # Convert bounding box to a NumPy array (required by ZED)
                box_points = np.array([
                    [x1, y1], [x2, y1],  # Top-left, Top-right
                    [x2, y2], [x1, y2]   # Bottom-right, Bottom-left
                ], dtype=np.float32)

                # 🔥 Track objects: Reuse instance_id if object is already seen
                centroid = (x1 + w // 2, y1 + h // 2)  # Use centroid for tracking
                existing_id = None

                for obj_id, prev_centroid in tracked_objects.items():
                    if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < 50:  # Adjust threshold if needed
                        existing_id = obj_id
                        break

                if existing_id is None:
                    instance_id = sl.generate_unique_id()
                else:
                    instance_id = existing_id

                tracked_objects[instance_id] = centroid  # Update tracked object position

                # Create and properly initialize the ZED custom box
                custom_box = sl.CustomBoxObjectData()
                custom_box.unique_object_id = instance_id  # ✅ Now correctly tracked
                custom_box.probability = conf * 100.0  # Convert to percentage
                custom_box.label = cls_id  # Integer class ID
                class_name = model.names[cls_id]
                custom_box.bounding_box_2d = box_points
                custom_box.is_grounded = False  # Change to True if objects are on the floor

                objects_in.append(custom_box)

            # Ingest custom detections into the ZED SDK tracking pipeline
            if objects_in:
                zed.ingest_custom_box_objects(objects_in)
            
            # Retrieve tracked objects from the ZED SDK
            objects = sl.Objects()
            obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
            obj_runtime_param.detection_confidence_threshold = 40
            zed.retrieve_objects(objects, obj_runtime_param)
            
            # Log and display each tracked object
            for obj in objects.object_list:
                log_object_data(obj, class_name)
                draw_detection_info(cv_image, obj, zed, class_name)
            
            cv2.imshow("ZED YOLO Object Detection", cv_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("\nESC pressed. Exiting...")
                running = False
                break

    print("\nCleaning up...")
    zed.close()
    cv2.destroyAllWindows()
    print("Program finished successfully. Now go kick some ass!")

if __name__ == "__main__":
    main()  
