import pyzed.sl as sl
import cv2
import numpy as np
import signal
import sys
import socket
import json
import time
from datetime import datetime

# Global flag for graceful shutdown
running = True

# Socket configuration
SERVER_IP = "192.168.138.179"  # Change to the IP address of the PC running arm_control.py
SERVER_PORT = 9999

# Socket client
client_socket = None

def setup_socket():
    """Establish connection to the arm control server"""
    global client_socket
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")
        return True
    except Exception as e:
        print(f"Socket connection failed: {str(e)}")
        return False

def send_object_data(position_mm, dimensions_mm, confidence):
    """Send object data to the arm control server"""
    global client_socket
    if client_socket is None:
        if not setup_socket():
            return False
    
    try:
        data = {
            "position": {
                "x": position_mm[0],
                "y": position_mm[1],
                "z": position_mm[2]
            },
            "dimensions": {
                "width": dimensions_mm[0],
                "height": dimensions_mm[1],
                "depth": dimensions_mm[2]
            },
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        # Convert to JSON and send
        json_data = json.dumps(data)
        client_socket.send((json_data + "\n").encode())
        print("Data sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send data: {str(e)}")
        # Attempt to reconnect
        try:
            client_socket.close()
        except:
            pass
        client_socket = None
        return False

def signal_handler(sig, frame):
    global running, client_socket
    print("\nShutdown signal received. Cleaning up...")
    running = False
    if client_socket:
        try:
            client_socket.close()
        except:
            pass

def safe_float_to_mm(value):
    """Safely convert float to millimeters handling all edge cases"""
    try:
        if np.isinf(value) or np.isnan(value):
            return 0
        mm_value = value * 1000  # Convert meters to millimeters
        if mm_value > 2147483647 or mm_value < -2147483648:  # 32-bit integer limits
            return 0
        return int(mm_value)
    except:
        return 0

def get_3d_position(point_cloud, x, y):
    """Get 3D position in millimeters with comprehensive error handling"""
    try:
        # Ensure coordinates are integers within bounds
        x = max(0, min(int(x), point_cloud.get_width()-1))
        y = max(0, min(int(y), point_cloud.get_height()-1))
        
        err, point3D = point_cloud.get_value(x, y, sl.MEM.CPU)
        if err != sl.ERROR_CODE.SUCCESS:
            return (0, 0, 0)
        
        # Convert each coordinate safely
        x_mm = safe_float_to_mm(point3D[0])
        y_mm = safe_float_to_mm(point3D[1])
        z_mm = safe_float_to_mm(point3D[2])
        
        return (x_mm, y_mm, z_mm)
    except:
        return (0, 0, 0)

def detect_green_cube(image):
    """Detect green cube with robust error handling"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        confidence = min(1.0, cv2.contourArea(largest_contour) / 10000)
        return (x, y, w, h), confidence, mask
    except:
        return None, None, None

def log_cube_data(position_mm, dimensions_mm):
    """Log cube information with validation"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print("\n" + "="*50)
        print(f"Timestamp: {timestamp}")
        print("\nPosition (mm):")
        print(f"  X: {position_mm[0]}")
        print(f"  Y: {position_mm[1]}")
        print(f"  Z: {position_mm[2]}")
        print("\nDimensions (mm):")
        print(f"  Width: {dimensions_mm[0]}")
        print(f"  Height: {dimensions_mm[1]}")
        print(f"  Depth: {dimensions_mm[2]}")
        print("="*50 + "\n")
    except:
        print("Error logging data")

def main():
    global running, client_socket
    
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    
    try:
        # Try to connect to server
        setup_socket()
        
        # Open camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open failed: {repr(err)}")
            return

        # Get camera resolution
        cam_info = zed.get_camera_information().camera_configuration
        width, height = cam_info.resolution.width, cam_info.resolution.height

        # Create display windows
        cv2.namedWindow("Green Cube Tracking", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detection Mask", cv2.WINDOW_NORMAL)
        
        print("\nStarting green cube tracking (mm units)...")
        print("Press ESC or Ctrl+C to exit")

        while running:
            try:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    # Get images and point cloud
                    left_image = sl.Mat()
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    point_cloud = sl.Mat()
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    
                    # Convert to OpenCV format
                    cv_image = left_image.get_data()
                    cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

                    # Detect green cube
                    cube_data, confidence, mask = detect_green_cube(cv_image_bgr)
                    
                    # Create default black mask if detection failed
                    if mask is None:
                        mask = np.zeros((height, width), dtype=np.uint8)
                    
                    if cube_data and confidence:
                        x, y, w, h = cube_data
                        centroid_x, centroid_y = x + w//2, y + h//2
                        
                        # Get 3D position in mm with full error handling
                        pos_x_mm, pos_y_mm, pos_z_mm = get_3d_position(point_cloud, centroid_x, centroid_y)
                        
                        # Only proceed if we have valid depth (Z > 0)
                        if pos_z_mm > 0:
                            # Get real-world dimensions
                            points = [
                                (x, y), (x + w, y), 
                                (x + w, y + h), (x, y + h)
                            ]
                            
                            # Get valid 3D coordinates
                            coords = []
                            for p in points:
                                coord = get_3d_position(point_cloud, p[0], p[1])
                                if coord[2] > 0:
                                    coords.append(coord)
                            
                            if len(coords) >= 2:
                                # Calculate dimensions in mm
                                width_mm = max(c[0] for c in coords) - min(c[0] for c in coords)
                                height_mm = max(c[1] for c in coords) - min(c[1] for c in coords)
                                depth_mm = sum(c[2] for c in coords) / len(coords)
                                
                                dimensions_mm = (abs(width_mm), abs(height_mm), abs(depth_mm))
                                
                                # Log cube data
                                log_cube_data((pos_x_mm, pos_y_mm, pos_z_mm), dimensions_mm)
                                
                                # Send data to arm control server
                                send_object_data((pos_x_mm, pos_y_mm, pos_z_mm), 
                                                dimensions_mm, 
                                                float(confidence))
                                
                                # Draw detection
                                cv2.rectangle(cv_image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(cv_image_bgr, f"X:{pos_x_mm}mm", (x, y-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(cv_image_bgr, f"Y:{pos_y_mm}mm", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.circle(cv_image_bgr, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    
                    # Display results
                    if mask.size > 0:
                        cv2.imshow("Detection Mask", mask)
                    cv2.imshow("Green Cube Tracking", cv_image_bgr)
                    
                    key = cv2.waitKey(10)
                    if key == 27:  # ESC key
                        running = False
            
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                continue

    finally:
        # Close socket connection
        if client_socket:
            try:
                client_socket.close()
                print("Socket connection closed")
            except:
                pass
                
        zed.close()
        cv2.destroyAllWindows()
        print("Tracking stopped.")

if __name__ == "__main__":
    main()
