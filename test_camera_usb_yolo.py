import cv2
import time
from ultralytics import YOLO
import urllib.request
import numpy as np
import time
import json


red_color = (0, 0, 255)


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
        self.zone_xy = []

class ZoneDetection:
    def __init__(self):
        """ playing zone """
        # Définir les 4 points du trapèze (x, y)
        self.pt1 = (200, 100)  # Haut-gauche
        self.pt2 = (600, 100)  # Haut-droit
        self.pt3 = (700, 400)  # Bas-droit
        self.pt4 = (100, 400)  # Bas-gauche
        self.color = red_color

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
        self.tracking_detection = []
        self.zone_detection = []

        self.time_start = None
        self.time_mean = []
        self.time_fps = ''

        self.key_plot = 0
        self.key_action = ''

    def box_detection_xy(self):
        """ compute xy in zone detection """

        for box_detection in self.box_detection:
            result = []
            xp = float((0.5 * (box_detection.x2 - box_detection.x1)) + box_detection.x1)
            yp = float(box_detection.y2)

            for zone_detection in self.zone_detection:
                Lx1 = float(zone_detection.pt2[0] - zone_detection.pt1[0])
                Lx2 = float(zone_detection.pt3[0] - zone_detection.pt4[0])
                Rx1 = (xp - zone_detection.pt1[0]) / Lx1
                Rx2 = (xp - zone_detection.pt4[0]) / Lx2
                Rx = 0.5 * (Rx1 + Rx2)

                Ly1 = zone_detection.pt4[1] - zone_detection.pt1[1]
                Ly2 = zone_detection.pt3[1] - zone_detection.pt2[1]
                Ry1 = (yp - zone_detection.pt1[1]) / Ly1
                Ry2 = (yp - zone_detection.pt2[1]) / Ly2
                Ry = 0.5 * (Ry1 + Ry2)

                x = (1.0 - Ry) * Rx1 + Ry * Rx2
                y = (1.0 - Rx) * Ry1 + Rx * Ry2

                result.append((x, y))

            box_detection.zone_xy = result

    def save_to_json(self, filepath="config.json"):
        """ Save config to file """
        data = {
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "zone_detection": []
        }
        for zone_detection in self.zone_detection:
            data["zone_detection"].append([
                list(zone_detection.pt1),
                list(zone_detection.pt2),
                list(zone_detection.pt3),
                list(zone_detection.pt4),
                list(zone_detection.color)])

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, filepath="config.json"):
        """Chargement des attributs depuis un fichier JSON"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.camera_width = data['camera_width']
        self.camera_height = data['camera_height']
        self.zone_detection = []
        for zone_detection in data['zone_detection']:
            new_zone_detection = ZoneDetection()
            new_zone_detection.pt1 = tuple(zone_detection[0])
            new_zone_detection.pt2 = tuple(zone_detection[1])
            new_zone_detection.pt3 = tuple(zone_detection[2])
            new_zone_detection.pt4 = tuple(zone_detection[3])
            new_zone_detection.color = tuple(zone_detection[4])
            self.zone_detection.append(new_zone_detection)

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

        zone_plots = self.get_zone_plot()
        cv2.circle(frame, zone_plots[self.key_plot], 10, red_color, 2)

        return frame

    def detect_camera(self):
        """ detect camera if needed, put USB number"""
        camera_usb_number = 0
        cap = cv2.VideoCapture(camera_usb_number)
        while not cap.isOpened():
            camera_usb_number += 1
            cap = cv2.VideoCapture(camera_usb_number)
            if camera_usb_number > 8:
                self.camera_usb_number = None
                break

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

    def get_zone_plot(self):
        """ return list of plot of zone detection """
        zone_plots = []
        for zone in self.zone_detection:
            zone_plots.append(zone.pt1)
            zone_plots.append(zone.pt2)
            zone_plots.append(zone.pt3)
            zone_plots.append(zone.pt4)
        return zone_plots

    def put_zone_plot(self, i_plot, plot):
        """ Put new plot on zone plot """
        i_zone = i_plot // 4
        i_plot = i_plot - i_zone * 4
        if i_plot == 0:
            self.zone_detection[i_zone].pt1 = plot
        elif i_plot == 1:
            self.zone_detection[i_zone].pt2 = plot
        elif i_plot == 2:
            self.zone_detection[i_zone].pt3 = plot
        elif i_plot == 3:
            self.zone_detection[i_zone].pt4 = plot

    def show_box_detection(self, frame):
        """ Add yolo box detection on frame """
        for bd in self.box_detection:
            cv2.rectangle(frame, (bd.x1, bd.y1), (bd.x2, bd.y2), (0, 255, 0), 2)
            cv2.putText(frame, bd.label, (bd.x1, bd.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            xp = int(0.5 * (bd.x2 - bd.x1)) + bd.x1
            yp = bd.y2
            cv2.circle(frame, (xp, yp), 10, red_color, 2)

            if bd.zone_xy:
                rx = int(bd.zone_xy[0][0] * 100.0)
                ry = int(bd.zone_xy[0][1] * 100.0)
                cv2.putText(frame, f"x: {rx} y: {ry}", (xp, yp + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 2)

        return frame

    def key_press(self, key):
        """ change box zone detection """
        if key == 32:
            # space press
            zone_plots = self.get_zone_plot()

            if self.key_plot == len(zone_plots) - 1:
                self.key_plot = 0
            else:
                self.key_plot += 1
        else:
            x = 0
            y = 0
            delta = 10
            if key == 82:  # flèche haut
                y -= delta
            elif key == 84:  # flèche bas
                y += delta
            elif key == 81:  # flèche gauche
                x -= delta
            elif key == 83:  # flèche droite
                x += delta

            if self.zone_detection and (x or y):
                zone_plots = self.get_zone_plot()
                new_plot = (zone_plots[self.key_plot][0] + x, zone_plots[self.key_plot][1] + y)
                self.put_zone_plot(self.key_plot, new_plot)

detect = CameraDetection()
detect.init_camera()
detect.init_model()
detect.load_from_json()


# Définir le codec et créer l'objet VideoWriter
# "XVID" => .avi, "mp4v" => .mp4
fourcc = cv2.VideoWriter_fourcc(*'XVID')
path = "/tmp/output_camera.avi"
out = cv2.VideoWriter(path, fourcc, 20.0, (detect.camera_width, detect.camera_height))



while True:
    # Capture une frame
    detect.get_camera_frame()
    detect.get_yolo_tracking()
    detect.box_detection_xy()

    # Dessiner les boîtes sur l'image originale
    out.write(detect.camera_frame)
    frame = detect.show_box_detection(detect.camera_frame)
    frame = detect.show_zone_detection(frame)

    # Affiche la frame
    cv2.imshow("Camera USB", frame)



    # Quitter avec la touche 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        detect.save_to_json()
        break
    elif key:
        detect.key_press(key)

# Libère les ressources
out.release()
cv2.destroyAllWindows()

