import cv2
import time
from ultralytics import YOLO
import urllib.request
import numpy as np
import time



class BoxDetection:
    def __init__(self,x1, y1, x2, y2, label, conf, class_id, track_id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
        self.conf = conf
        self.class_id = class_id # Yolo classification 0 = personne
        self.track_id = track_id # Yolo tracking 0 = None

class ZoneDetection:
    def __init__(self):
        """ playing zone """
        # Définir les 4 points du trapèze (x, y)
        self.pt1 = (200, 100)  # Haut-gauche
        self.pt2 = (600, 100)  # Haut-droit
        self.pt3 = (700, 400)  # Bas-droit
        self.pt4 = (100, 400)  # Bas-gauche
        self.color = (0, 0, 255)

    def show_zone_detection(self, frame):
        """ Add polyline on frame """
        trapeze_pts = np.array([self.pt1, self.pt2, self.pt3, self.pt4], dtype=np.int32)
        trapeze_pts = trapeze_pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [trapeze_pts], isClosed=True, color=self.color, thickness=2)
        return frame


class CameraDetection:
    def __init__(self):
        self.camera_usb_number = None
        self.camera = None
        self.camera_width = 1280
        self.camera_height = 800
        self.camera_frame = None
        # resolution: 1280 800 FPS: 11
        # resolution: 640 480: FPS: 33
        # resolution: 320 240 FPS: 200

        # Réglages possibles (valeurs à adapter à ta caméra)
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)   # Luminosité (0.0 à 1.0 ou plage spécifique)
        # cap.set(cv2.CAP_PROP_CONTRAST, 0.5)     # Contraste
        # cap.set(cv2.CAP_PROP_SATURATION, 0.5)   # Saturation
        # cap.set(cv2.CAP_PROP_HUE, 0.5)          # Teinte
        # cap.set(cv2.CAP_PROP_SHARPNESS, 0.5)    # Netteté (pas toujours pris en charge)
        # cap.set(cv2.CAP_PROP_GAMMA, 0.5)        # Gamma
        # cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4500)  # Balance des blancs (Kelvin)
        # cap.set(cv2.CAP_PROP_BACKLIGHT, 1)      # Compensation du rétroéclairage
        # cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # Exposition (souvent négatif = automatique désactivé)

        self.yolo_model_name = "yolo11m.pt"
        self.yolo_conf = 0.5 # seuil de confiance
        self.yolo_filter_class = [] # 0: personne, list of class to track
        self.yolo_model = None
        self.box_detection = []
        self.zone_detection = []

        self.time_start = None
        self.time_mean = []
        self.time_fps = ''


    def track_fps(self, nb_time=10):
        """ track FPS """
        time_start = time.time()
        if self.time_start is not None:
            time_mean = time_start - self.time_start
            self.time_mean.append(time_mean)

        if len(self.time_mean) > nb_time:
            del(self.time_mean[0])
            fps = int(1.0 / (sum(time_mean) / len(time_mean)))
            self.time_fps = f"{fps}"

    def load_zone_detection(self):
        """ load zone detection """
        zone_detection = ZoneDetection()
        self.zone_detection.append(zone_detection)

    def show_zone_detection(self, frame):
        """ add zone detection on frame """
        for zone_detection in self.zone_detection:
            frame = zone_detection.show_zone_detection(frame)
        return frame

    def detect_camera(self):
        """ detect camera if needed, put USB number"""
        camera_usb_number = 0
        cap = cv2.VideoCapture(camera_usb_number)
        while not cap.isOpened():
            camera_usb_number += 1
            cap = cv2.VideoCapture(camera_usb_number)
            if camera_usb_number > 6:
                self.camera_usb_number = None

        self.camera_usb_number = camera_usb_number

    def init_camera(self):
        """ open camera """
        if self.camera_usb_number is None:
            self.detect_camera()
        self.camera = cv2.VideoCapture(self.camera_usb_number)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

    def end_camera(self):
        """ resource free """
        self.camera.release()

    def init_model(self):
        """ Load a yolo model """
        self.yolo_model = YOLO(self.yolo_model_name)

    def get_camera_frame(self):
        """ Return the frame of the camera """
        ret, frame = self.camera.read()
        if ret:
            self.camera_frame = frame
        else:
            # error
            pass

    def get_yolo_tracking(self):
        """ return yolo tracking """
        self.box_detection = []
        results = self.yolo_model.track(self.camera_frame, conf=self.yolo_conf, persist=True)
        detections = results[0].boxes.xyxy  # Coordonnées [x1, y1, x2, y2]
        classes = results[0].boxes.cls  # Indices des classes détectées
        confs = results[0].boxes.conf  # Niveaux de confiance
        track_ids = results[0].boxes.id

        # sauvegarder les boîtes detectées
        for i, box in enumerate(detections):

            class_id = int(classes[i])
            # Filtrer pour ne garder que les personnes
            if self.yolo_filter_class and class_id not in self.yolo_filter_class:
                continue

            x1, y1, x2, y2 = map(int, box)  # Conversion en int
            conf = float(confs[i])
            if track_ids is not None:
                track_id = int(track_ids[i])
            else:
                track_id = 0

            class_name = self.yolo_model.names[class_id]
            label = f"{track_id} - {class_name} {conf:.2f}"

            new_track =  BoxDetection(x1, y1, x2, y2, label, conf, class_id, track_id)
            self.box_detection.append(new_track)


    def show_box_detection(self, frame):
        """ Add yolo box detection on frame """
        for bd in self.box_detection:
            cv2.rectangle(frame, (bd.x1, bd.y1), (bd.x2, bd.y2), (0, 255, 0), 2)
            cv2.putText(frame, bd.label, (bd.x1, bd.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame




detect = CameraDetection()
detect.init_camera()
detect.init_model()
detect.load_zone_detection()

while True:
    # Capture une frame
    detect.get_camera_frame()
    detect.get_yolo_tracking()


    # Dessiner les boîtes sur l'image originale
    frame = detect.show_box_detection(detect.camera_frame)
    frame = detect.show_zone_detection(frame)

    # Affiche la frame
    cv2.imshow("Camera USB", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère les ressources
cv2.destroyAllWindows()

