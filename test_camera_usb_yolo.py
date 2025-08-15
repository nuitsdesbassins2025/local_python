import cv2
import time
from ultralytics import YOLO
import urllib.request
import numpy as np
import time
import cv2

#model yolo11 n, m, l, x,
model_name = "yolo11m.pt"
model = YOLO(model_name)
# Détection avec YOLO




# Ouvre la caméra (0 = première caméra USB trouvée)
camera_usb_number = 2
cap = cv2.VideoCapture(camera_usb_number)
# Set the resolution if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)


# Réglages possibles (valeurs à adapter à ta caméra)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)   # Luminosité (0.0 à 1.0 ou plage spécifique)
#cap.set(cv2.CAP_PROP_CONTRAST, 0.5)     # Contraste
#cap.set(cv2.CAP_PROP_SATURATION, 0.5)   # Saturation
#cap.set(cv2.CAP_PROP_HUE, 0.5)          # Teinte
#cap.set(cv2.CAP_PROP_SHARPNESS, 0.5)    # Netteté (pas toujours pris en charge)
#cap.set(cv2.CAP_PROP_GAMMA, 0.5)        # Gamma
#cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4500)  # Balance des blancs (Kelvin)
#cap.set(cv2.CAP_PROP_BACKLIGHT, 1)      # Compensation du rétroéclairage
#cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # Exposition (souvent négatif = automatique désactivé)


if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()
# Get the actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('resolution:', width, height)

#resolution: 1280 800 FPS: 0.09310606414196539 min  0.08899354934692383 - max 0.0979154109954834
#resolution: 640 480: FPS: 0.028238801395191866 min  0.025682449340820312 - max 0.03220987319946289
#resolution: 320 240 FPS: 0.005150107776417452 min - max 0.0025167465209960938 0.00950312614440918

time_mean = []

while True:
    # Capture une frame
    time_start = time.time()
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur : Impossible de lire la frame.")
        break


    
    results = model(frame, conf=0.5)  # conf = seuil de confiance
    
    detections = results[0].boxes.xyxy  # Coordonnées [x1, y1, x2, y2]
    classes = results[0].boxes.cls      # Indices des classes détectées
    confs = results[0].boxes.conf       # Niveaux de confiance

    # Dessiner les boîtes sur l'image originale
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)  # Conversion en int
        cls_id = int(classes[i])
        conf = float(confs[i])

        # Filtrer pour ne garder que les personnes (classe 0 dans COCO)

        
            
        class_name = model.names[cls_id] 
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
    # Afficher le résultat
    #cv2.imshow("Détection de personnes - YOLO", annotated_frame)
    # Affiche la frame
    cv2.imshow("Camera USB", frame)
    
    time_end = time.time()
    time_mean.append(time_end - time_start)
    if len(time_mean) > 50:
        fps_ms = sum(time_mean) / len(time_mean)
        print('FPS:', fps_ms, 'min - max', min(time_mean), max(time_mean))
        time_mean = []
    

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère les ressources
cap.release()
cv2.destroyAllWindows()

