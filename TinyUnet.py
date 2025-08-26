import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import threading
import time
import websocket

#ESP32 WebSocket Setup
esp32_host = "ws://esp32team4.local:82"
latest_command = "X"  # Start with safe stop
lock = threading.Lock()

def send_command_loop():
    ws = None
    while True:
        try:
            if ws is None:
                ws = websocket.create_connection(esp32_host)
                print("Connected to ESP32")

            with lock:
                command = latest_command

            ws.send(f"{command}")
            print(f"Sent to ESP32:{command}")

            time.sleep(0.1)  # Send command every 100ms

        except Exception as e:
            print(f"ESP32 send error: {e}")
            try:
                if ws:
                    ws.close()
            except:
                pass
            ws = None
            time.sleep(1)

#TinyUNet Model Definition
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        m = self.middle(self.pool2(d2))
        u2 = self.upconv2(torch.cat([self.up2(m), d2], dim=1))
        u1 = self.upconv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out(u1)

#Load the model
#update the path
model_path = r'C:\Users\stsaa\Desktop\project\tinyunet_model.pth'
model = TinyUNet(in_ch=3, out_ch=3)
model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

#Transforms
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

#Inference function
def infer_frame(frame):
    h, w, _ = frame.shape
    masked_frame = frame.copy()
    masked_frame[0:h//2, :] = 0  # Mask top half

    pil_image = Image.fromarray(cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB))
    resized = transform(pil_image).unsqueeze(0)
    if torch.cuda.is_available():
        resized = resized.cuda()

    with torch.no_grad():
        output = model(resized)[0]
        pred = output.argmax(0).cpu().numpy()
        conf = torch.nn.functional.softmax(output, dim=0).max(0).values.cpu().numpy()

    return pred, conf

#Red & Blue mask visualization
def mask_only_frame(pred_mask):
    h, w = pred_mask.shape
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)

    mask_vis[pred_mask == 2] = [255, 0, 0]  # Blue lanes

    center_x = w // 2
    center_y = h // 2
    q4_top = int(center_y + (0.3 * (h - center_y)))

    cv2.line(mask_vis, (center_x, 0), (center_x, h), (255, 255, 255), 2)
    cv2.line(mask_vis, (0, center_y), (w, center_y), (255, 255, 255), 2)
    cv2.line(mask_vis, (center_x, q4_top), (w, q4_top), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(mask_vis, "1", (center_x//2 - 10, center_y//2), font, font_scale, color, thickness)
    cv2.putText(mask_vis, "2", (center_x + center_x//2 - 10, center_y//2), font, font_scale, color, thickness)
    cv2.putText(mask_vis, "3", (center_x//2 - 10, center_y + (h - center_y)//2), font, font_scale, color, thickness)
    cv2.putText(mask_vis, "4a", (center_x + center_x//2 - 20, center_y + int(0.15 * (h - center_y))), font, font_scale, color, thickness)
    cv2.putText(mask_vis, "4b", (center_x + center_x//2 - 20, q4_top + int(0.35 * (h - center_y))), font, font_scale, color, thickness)

    return mask_vis

#Start ESP32 communication thread
threading.Thread(target=send_command_loop, daemon=True).start()

#Visualization Toggl
SHOW_OUTPUT = True  # Set to False to disable cv2.imshow windows

#Video Processing
video_url = "http://localhost:8000/video_feed"
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    pred_mask, conf_mask = infer_frame(frame_resized)

    h, w = pred_mask.shape
    center_x, center_y = w // 2, h // 2
    q4_top = int(center_y + 0.3 * (h - center_y))

    q3 = pred_mask[center_y:, :center_x]
    q4a = pred_mask[center_y:q4_top, center_x:]
    q4b = pred_mask[q4_top:, center_x:]

    q3_blue = np.sum(q3 == 2)
    q4a_blue = np.sum(q4a == 2)
    q4b_blue = np.sum(q4b == 2)

    if q4b_blue > 100:
        movement = "D"  # Forward
    elif q3_blue > 500 or (q3_blue + q4a_blue > 500):
        movement = "W"  # Right
    else:
        movement = "A"  # Left

    with lock:
        latest_command = movement

    if SHOW_OUTPUT:
        mask_output = mask_only_frame(pred_mask)
        cv2.putText(mask_output, f"Move: {movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Lane Mask", mask_output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
