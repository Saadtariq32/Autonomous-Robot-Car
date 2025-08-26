import cv2 
import socket
import struct
import time

use_camera = True  # False to test with an image 
frame_width = 640

WINDOWS_HOSTNAME = "127.0.0.1"
TCP_PORT = 5005

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect_tcp():
    while True:
        try:
            print(f"[TCP] Connecting to {WINDOWS_HOSTNAME}:{TCP_PORT}...")
            tcp_socket.connect((WINDOWS_HOSTNAME, TCP_PORT))
            print("[TCP] Connected to PC.")
            break
        except Exception:
            print("[TCP] Connection refused, retrying...")
            time.sleep(2)

connect_tcp()


def send_frame_with_command(sock, frame, command_str="default"):
    try:
        if not sock or sock.fileno() == -1:
            print("[TCP] Socket is closed. Reconnecting...")
            reconnect_tcp()

        _, buffer = cv2.imencode('.jpg', frame)
        img_data = buffer.tobytes()
        command_bytes = command_str.encode()

        total_len = 4 + len(command_bytes) + len(img_data)
        command_len = len(command_bytes)

        header = struct.pack('>LL', total_len, command_len)
        packet = header + command_bytes + img_data

        sock.sendall(packet)
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"[TCP] Connection error: {e}. Attempting to reconnect.")
        reconnect_tcp()
    except Exception as e:
        print(f"[TCP] Send error: {e}")

def reconnect_tcp():
    global tcp_socket
    print("[TCP] Reconnecting to PC...")
    try:
        if tcp_socket:
            tcp_socket.close()
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connect_tcp()
    except Exception as e:
        print(f"[TCP] Reconnection failed: {e}")
        time.sleep(2)
        reconnect_tcp()

if use_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        send_frame_with_command(tcp_socket, frame)

    cap.release()
else:
    image_path = 'test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
    else:
        send_frame_with_command(tcp_socket, image)
