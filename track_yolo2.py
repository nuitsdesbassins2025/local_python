from ultralytics import YOLO
import urllib.request
import numpy as np
import time
import cv2

#model yolo11 n, m, l, x,
model_name = "yolo11m.pt"


model = YOLO(model_name)
ip_camera = "192.168.0.225"
CAMERA_URL = f"rtsp://admin:1234567890@{ip_camera}:554/channel=0_stream=1&onvif=0.sdp?real_stream="

CAMERA_URL = f'http://{ip_camera}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=test&user=admin&password=lndb2025'
CAMERA_URL = f'rtmp://admin:lndb2025@{ip_camera}:1935/live/stream'
#rtmp://admin:lndb2025@192.168.0.225:1935/live/stream
#VIDEO_SOURCE = '/home/joannes/Vidéos/EarthCam_live_dublin02.mp4' #https://www.youtube.com/watch?v=u4UZ4UvZXrg,
VIDEO_SOURCE = CAMERA_URL

while True:
    time_start = time.time()
    img = urllib.request.urlopen(CAMERA_URL, timeout=1)
    img_array = np.array(bytearray(img.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_array, -1)
    path = f'/tmp/fram{time.time()}.jpg'.replace(' ', '-')
    cv2.imwrite(path , frame)
    #cv2.imshow('Camera Stream', frame)
    key = cv2.waitKey(1)
    print('temps', time.time() - time_start)

    

    
    
    
