import struct
import cv2
import numpy as np
import time
import torch
import threading
import queue
import pathlib
from flask import Flask, Response, render_template
from joblib import load
import urllib.request, json
import pandas as pd
import websocket
import os
import socket

# TCP and YOLO setup
TCP_IP = "0.0.0.0"
TCP_PORT = 5005
ESP32_HOSTNAME = "esp32team4.local"
ESP32_PORT = 81

pathlib.PosixPath = pathlib.WindowsPath
#Update the path 
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\stsaa\Desktop\project\yolov5n.pt', force_reload=True)
model.conf = 0.5

#Globals
frame_queue = queue.Queue(maxsize=25)
latest_frame = None
detected_object = ""
weather_condition = "Checking..."
emoji_map = {
    "santa": "🎅",
    "moose": "🦌",
    "bear": "🐻",
    "fox": "🦊"
}
weather_emoji_map = {
    "Snow": "❄️",
    "Rain": "🌧️",
    "Normal": "☀️",
    "Ice": "🧊"
}

object_speed_percent = 100
weather_speed_percent = 100
total_speed_percent = 100

#Flask setup
app = Flask(__name__)
ws = None

#TCP Receiver
def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("Socket closed before receiving all data.")
        data += more
    return data

def receiver_thread():
    global latest_frame
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(1)
    print("TCP Receiver started on port", TCP_PORT)

    conn, addr = sock.accept()
    print(f"Connected to: {addr}")

    try:
        while True:
            header = recvall(conn, 8)
            total_len, command_len = struct.unpack('>LL', header)
            _ = recvall(conn, command_len)
            img_data = recvall(conn, total_len - 4 - command_len)

            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                latest_frame = cv2.resize(img, (640, 480))
                if not frame_queue.full():
                    frame_queue.put(latest_frame)
    except Exception as e:
        print(f"[Receiver Error] {e}")
        conn.close()

#YOLO Inference Thread
def inference_thread():
    global latest_frame, detected_object, object_speed_percent
    while True:
        if latest_frame is not None:
            results = model(latest_frame)
            found = set()

            for *box, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)].lower()
                if label in emoji_map:
                    found.add(label)

            if found:
                detected_object = ' '.join(f"{emoji_map[label]} {label.capitalize()}" for label in sorted(found))
                object_speed_percent = 80
            else:
                detected_object = ""
                object_speed_percent = 100
        time.sleep(0.03)

#Weather Prediction Thread
def weather_thread():
    global weather_condition, weather_speed_percent
    labels = ["Ice", "Normal", "Rain", "Snow"]
    model = load(r"C:\Users\stsaa\Desktop\project\weathercontrolmodel.joblib") #Update the path 

    while True:
        try:
            with urllib.request.urlopen("https://edu.frostbit.fi/api/road_weather/2025/") as url:
                data = json.load(url)

            if data:
                df = pd.DataFrame([data])
                prediction = labels[model.predict(df)[0]]
                emoji = weather_emoji_map.get(prediction, "")
                weather_condition = f"{emoji} {prediction}"

                if prediction == "Snow":
                    weather_speed_percent = 80
                elif prediction == "Rain":
                    weather_speed_percent = 90
                elif prediction == "Ice":
                    weather_speed_percent = 70
                else:
                    weather_speed_percent = 100
        except Exception as e:
            print(f"[Weather Error] {e}")
            weather_condition = "⚠️ Error"
            weather_speed_percent = 100
        
        time.sleep(60)

#WebSocket Client
def websocket_thread():
    global total_speed_percent, ws
    while True:
        try:
            if ws is None or not ws.connected:
                ws = websocket.WebSocket()
                ws.connect(f"ws://{ESP32_HOSTNAME}:{ESP32_PORT}")
                print("Connected to ESP32 WebSocket")

            # Base speed
            base_speed = 100
            object_reduction = 20 if object_speed_percent < 100 else 0
            weather_reduction = 0

            if weather_speed_percent == 90:
                weather_reduction = 10
            elif weather_speed_percent == 80:
                weather_reduction = 20
            elif weather_speed_percent == 70:
                weather_reduction = 30

            total_speed_percent = max(50, base_speed - (object_reduction + weather_reduction))

            ws.send(f"SPEED:{total_speed_percent}")
        except Exception as e:
            print(f"[WebSocket Error] {e}")
            ws = None
        time.sleep(1)

#MJPEG Streaming
def generate_video_stream():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

#Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected')
def get_detected():
    return f"⚠️ {detected_object} ahead - Speed reduced 20%" if detected_object else "None"

@app.route('/weather')
def get_weather():
    return weather_condition

@app.route('/object_speed')
def object_speed():
    return str(object_speed_percent)

@app.route('/weather_speed')
def weather_speed():
    return str(weather_speed_percent)

@app.route('/total_speed')
def total_speed():
    return str(total_speed_percent)

#Run App
if __name__ == '__main__':
    threading.Thread(target=receiver_thread, daemon=True).start()
    threading.Thread(target=inference_thread, daemon=True).start()
    threading.Thread(target=weather_thread, daemon=True).start()
    threading.Thread(target=websocket_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=8000, debug=False)