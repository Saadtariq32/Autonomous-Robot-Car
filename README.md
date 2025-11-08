# Autonomous Robot Car  
Repository for an autonomous vehicle platform built using Raspberry Pi, ESP32, and camera-based perception.

## Project Overview  
This project implements an autonomous vehicle that uses a Raspberry Pi and ESP32 as the primary hardware stack.  
The system processes camera input using:
- TinyUNet (lane segmentation)
- YOLOv5 (object detection)
- Random Forest (weather-based speed prediction)

These models collectively determine movement decisions that are executed by the ESP32 through a motor driver.

## Key Features  
- Computer Vision on Raspberry Pi for lane detection and object recognition  
- ESP32 microcontroller for movement control via L298N motor driver  
- Flask-based REST API backend for remote monitoring and communication  
- YOLOv5 object detection to adjust navigation and speed  
- TinyUNet lane segmentation to keep the car within marked tracks  
- Random Forest weather prediction to dynamically adjust speed  
- Video pipeline + UI for live feedback and debugging  

## System Pipeline  

1. Raspberry Pi captures live video.  
2. Video is forwarded via TCP to the Flask backend.  
3. Flask backend sends frames to the YOLOv5 inference thread, which:  
   - Detects objects  
   - Annotates results for the Web UI  
   - Reduces speed if certain classes are detected  
4. The same frames are passed to the TinyUNet lane segmentation thread, which:  
   - Identifies track boundaries  
   - Generates steering guidance  
5. Weather data is passed through a Random Forest model to adjust speed.  
6. A final movement command (direction + velocity) is sent to the ESP32.  
7. ESP32 controls motors → vehicle drives autonomously.  

![Robot Car](https://github.com/user-attachments/assets/e4f56a5a-24dc-4fba-8c33-c238ae73571b)
