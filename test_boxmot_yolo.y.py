import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from ultralytics import YOLO  # <-- YOLOv8
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolo12m.pt')  # ou 'yolov8n.pt', 'yolov8m.pt', ...
model.to(device)

# Initialize tracker
tracker = BoostTrack(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  # chemin vers ton modèle ReID
    device=device,
    half=True  # utilise half precision si tu veux (True pour GPU)
)

# Start video capture
cap = cv2.VideoCapture('/home/joannes/Vidéos/nuitsdesbassins/output_camera_03.avi')

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference
    # results est une liste de Results objects (un par image, ici une seule)
    results = model(frame, verbose=False, device=device, conf=0.5)  # conf threshold ici

    # Extraire les détections
    detections = []
    for r in results:
        boxes = r.boxes  # Boîtes de détection
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # coordonnées
            conf = box.conf[0].cpu().numpy()             # confiance
            cls = box.cls[0].cpu().numpy()               # classe
            detections.append([x1, y1, x2, y2, conf, cls])

    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
    print('-------detections-------', detections)

    tracker_time = time.time()
    # Update tracker
    #   INPUT:  M X (x1, y1, x2, y2, conf, cls)
    #   OUTPUT: M X (x1, y1, x2, y2, id, conf, cls, ind)
    tracked_objects = tracker.update(detections, frame)
    print('------tracked_objects-----', time.time() - tracker_time, '\n', tracked_objects)

    # Dessiner les résultats du tracker
    tracker.plot_results(frame, show_trajectories=True)

    # Afficher l'image
    cv2.imshow('BoXMOT + YOLOv8', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    while True and key != ord('a'):
        key = cv2.waitKey(1) & 0xFF

# Clean up
cap.release()
cv2.destroyAllWindows()