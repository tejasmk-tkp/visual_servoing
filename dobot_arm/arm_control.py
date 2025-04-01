import numpy as np
from pydobot import Dobot
import time
import socket
import json
import threading
import signal
import sys

# Socket Server Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 9999

# Queue for cube positions from socket
detected_objects = []
object_lock = threading.Lock()
server_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global server_running
    print("\nShutdown signal received. Cleaning up...")
    server_running = False
    sys.exit(0)

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_socket_server():
    """Start a socket server to receive object data"""
    global server_running, detected_objects
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        server_socket.settimeout(1)  # Add timeout to allow checking server_running flag
        print(f"Socket server started on {HOST}:{PORT}")
        
        while server_running:
            try:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address} established")
                client_handler = threading.Thread(
                    target=handle_client,
                    args=(client_socket, address)
                )
                client_handler.daemon = True
                client_handler.start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Socket accept error: {str(e)}")
                time.sleep(1)
    
    except Exception as e:
        print(f"Socket server error: {str(e)}")
    finally:
        server_socket.close()
        print("Socket server closed")

def handle_client(client_socket, address):
    """Handle incoming client connections"""
    global detected_objects, object_lock
    
    buffer = ""
    try:
        print(f"Handling connection from {address}")
        while server_running:
            # Receive data in chunks
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                break
                
            buffer += data
            
            # Process complete JSON messages
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    object_data = json.loads(line)
                    print(f"Received object data: {object_data}")
                    
                    # Convert position to format expected by arm control
                    position = (
                        object_data['position']['x'],
                        object_data['position']['y'],
                        object_data['position']['z']
                    )
                    
                    # Add to queue with thread safety
                    with object_lock:
                        detected_objects.append(position)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {line}")
    
    except Exception as e:
        print(f"Client handler error: {str(e)}")
    finally:
        client_socket.close()
        print(f"Connection from {address} closed")

# Connect to Dobot
def connect_dobot():
    """Connect to the Dobot and return device object"""
    try:
        port = "COM8"  # Change according to your system
        print(f"Connecting to Dobot on port {port}...")
        device = Dobot(port=port)
        print("Dobot connected successfully")
        return device
    except Exception as e:
        print(f"Error connecting to Dobot: {str(e)}")
        return None

def dh_transform(theta, alpha, d, a):
    """Computes the DH transformation matrix."""
    theta = np.radians(theta)  # Convert to radians
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# DH Parameters (theta in degrees, alpha in degrees, d in mm, a in mm)
dh_params = [
    (0, -90, 8, 0),    # Link 1
    (0, 0, 0, 135),    # Link 2
    (0, 0, 0, 147),    # Link 3
    (0, -90, 0, 59.7)  # Link 4 (End-Effector)
]

# Xc and Zc in home position (to be set manually)
xc_home = 50.0  # Example value, update as needed
zc_home = 30.0  # Example value, update as needed

# Workspace Limits
workspace_limits = {
    "x_min": 110, "x_max": 327,
    "y_min": -309, "y_max": 309,
    "z_min": -132, "z_max": 159
}

# Joint Limits
j1_min, j1_max = -124, 106
j2_min, j2_max = -4.2, 90
j3_min, j3_max = -10, 74

def get_j3_angle(device):
    """Fetch the J3 angle from Dobot API"""
    pose_data = device.pose()
    return pose_data[4] if len(pose_data) >= 5 else 0

def get_xc_zc(j3_angle, xc_home, zc_home):
    """Compute Xc and Zc dynamically based on J3 angle."""
    xc = xc_home * np.cos(np.radians(j3_angle))  # Xc calculation
    zc = zc_home + xc_home * np.sin(np.radians(j3_angle))  # Zc calculation
    return xc, zc

def compute_base_to_camera(xc, zc, j3_angle):
    """Compute the Base-to-Camera Transformation (T_BC)"""
    
    # Compute Base-to-End-Effector transformation (T_BE)
    T_BE = np.eye(4)
    for theta, alpha, d, a in dh_params:
        T_BE = np.dot(T_BE, dh_transform(theta, alpha, d, a))
    
    # Updated End-Effector to Camera Transformation (T_EC)
    T_EC = np.array([
        [np.cos(np.radians(j3_angle)), 0, np.sin(np.radians(j3_angle)), xc],
        [0, 1, 0, 0],
        [-np.sin(np.radians(j3_angle)), 0, np.cos(np.radians(j3_angle)), zc],
        [0, 0, 0, 1]
    ])
    
    return np.dot(T_BE, T_EC)

def transform_object_to_base(object_coords, T_BC):
    """Transforms object coordinates from camera frame to base frame."""
    obj_cam = np.array([object_coords[0], object_coords[1], object_coords[2], 1])
    obj_base = np.dot(T_BC, obj_cam)
    return obj_base[:3]

def is_within_workspace(x, y, z):
    """Check if coordinates are within workspace limits."""
    return (workspace_limits["x_min"] <= x <= workspace_limits["x_max"] and
            workspace_limits["y_min"] <= y <= workspace_limits["y_max"] and
            workspace_limits["z_min"] <= z <= workspace_limits["z_max"])

def scan_for_object(device):
    """Rotate J1 to scan for objects."""
    for j1_angle in range(j1_min, j1_max + 1, 10):  # Rotate in steps
        device.move_to(259.1, 0, -8.3, j1_angle, wait=True)
        time.sleep(1)
        
        # Check if we have received any object positions from the socket
        with object_lock:
            if detected_objects:
                object_coords_camera = detected_objects.pop(0)
                return object_coords_camera
                
    return None

def main():
    global server_running, detected_objects, object_lock
    
    # Start socket server thread
    server_thread = threading.Thread(target=start_socket_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Allow some time for server to start
    time.sleep(1)
    
    # Connect to Dobot
    device = connect_dobot()
    if not device:
        print("Failed to connect to Dobot. Exiting.")
        server_running = False
        return
    
    try:
        # Start from Home Position
        print("Moving to Home Position...")
        device.move_to(259.1, 0, -8.3, 0, wait=True)
        time.sleep(2)
        
        # Main control loop
        while server_running:
            # First check if we have received object data from socket
            object_coords_camera = None
            with object_lock:
                if detected_objects:
                    object_coords_camera = detected_objects.pop(0)
            
            # If no object received from socket, scan for one
            if not object_coords_camera:
                print("No object data received from socket. Scanning...")
                object_coords_camera = scan_for_object(device)
            
            if object_coords_camera:
                print(f"Processing object at camera coordinates: {object_coords_camera}")
                j3_angle = get_j3_angle(device)
                xc, zc = get_xc_zc(j3_angle, xc_home, zc_home)
                T_BC = compute_base_to_camera(xc, zc, j3_angle)
                object_coords_base = transform_object_to_base(object_coords_camera, T_BC)
                
                if is_within_workspace(*object_coords_base):
                    print("Object detected within workspace. Moving to pick it up...")
                    device.move_to(object_coords_base[0], object_coords_base[1], object_coords_base[2], 0, wait=True)
                    time.sleep(1)
                    device.suck(True)
                    time.sleep(5)
                    device.move_to(259.1, 0, -8.3, 0, wait=True)
                    device.suck(False)
                else:
                    print("Object is out of workspace bounds. Ignoring.")
            else:
                print("No object detected. Waiting...")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down...")
    except Exception as e:
        print(f"Error in main control loop: {str(e)}")
    finally:
        # Clean up
        if device:
            device.close()
        server_running = False
        print("Process complete.")

if __name__ == "__main__":
    main()
