import cv2
import time
from ultralytics import YOLO
import urllib.request
import numpy as np
import time
import json
import httpx
import multiprocessing
import torch
from boxmot import BoostTrack
from boxmot import ByteTrack
from pathlib import Path
import threading
import argparse


red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)

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
        self.track_boost_id = 0 # BoostTrack tracking 0 = None
        self.track_byte_id = 0  # ByteTrack tracking 0 = None



class TrackingDetection:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.last_position = []
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.mean_w = []
        self.mean_y = []
        self.x1_ok = 0
        self.y1_ok = 0
        self.x2_ok = 0
        self.y2_ok = 0
        self.occluded_by_box = []
        self.show_last_position = []
        self.label = 'new'
        self.class_id = 0 # Yolo classification 0 = personne

        self.track_id = 0 # Yolo tracking 0 = None
        self.track_boost_id = 0 # BoostTrack tracking 0 = None
        self.track_byte_id = 0  # ByteTrack tracking 0 = None

        self.track_ids = []
        self.track_boost_ids = []
        self.track_byte_ids = []

        self.tracking_id = 0
        self.related_client_id = ''
        self.lost_frame = 0
        self.zone_xy = []
        self.state = 'new'
        self.conf = 0
        self.tracker = None

    def update_by_boxdetection(self, boxdetection):
        """ update value """
        self.x1 = boxdetection.x1
        self.y1 = boxdetection.y1
        self.x2 = boxdetection.x2
        self.y2 = boxdetection.y2

        self.x1_ok = boxdetection.x1
        self.y1_ok = boxdetection.y1
        self.x2_ok = boxdetection.x2
        self.y2_ok = boxdetection.y2

        self.class_id = boxdetection.class_id
        self.conf = boxdetection.conf

        self.track_id = boxdetection.track_id
        self.track_boost_id = boxdetection.track_boost_id
        self.track_byte_id = boxdetection.track_byte_id

        self.tracker = None

    def intersection_aera(self, trackingbox):
        """ return intersection aera """
        xA = max(self.x1, trackingbox.x1)
        yA = max(self.y1, trackingbox.y1)
        xB = min(self.x2, trackingbox.x2)
        yB = min(self.y2, trackingbox.y2)

        if xB <= xA or yB <= yA:
            return 0.0  # pas d'intersection

        return (xB - xA) * (yB - yA)


    def get_bbox(self):
        """ return bbox """
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def update_by_opencvtracking(self, frame, previews_frame):
        """ Complete lost tracking by opencv tracker """
        bbox = (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
        if self.tracker is None:
            self.tracker = cv2.legacy.TrackerCSRT_create()

        self.tracker.init(previews_frame, bbox)
        success, bbox = self.tracker.update(frame)
        if success:
            (x1, y1, w, h) = [int(v) for v in bbox]
            self.x1 = x1
            self.y1 = y1
            self.x2 = x1 + w
            self.y2 = y1 + h

    def get_visible_y2(self):
        """ return bbox visible from x1,y1 """
        box = (self.x1, self.y1, self.x2, self.y2)
        for xybox in self.occluded_by_box:
            if xybox[1] > box[3]:
                box[3] = xybox[1]
        if box[1] > box[3]:
            box[3] = box[1]
        bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
        return bbox

    def tread_opencvtracking(self, frame, previews_frame, queue):
        """ Complete lost tracking by opencv tracker """
        res = (self.x1, self.y1, self.x2, self.y2)
        w_origin = self.x2 - self.x1
        h_origin = self.y2 - self.y1
        bbox = self.get_visible_y2()
        # Check 20% of visibility
        if bbox[3] > 0.2 * h_origin:
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(previews_frame, bbox)
            success, bbox = tracker.update(frame)
            if success:
                (x1, y1, w, h) = [int(v) for v in bbox]
                res = (x1, y1, x1 + w_origin, y1 + h_origin)
        queue.put(res)

    def intersection_boxdetection(self, boxdetection):
        """ return % of surface with boxdetection"""
        surface = (self.x2 - self.x1) * (self.y2 - self.y1)
        if surface != 0.0:
            inter_w = max(0, min(self.x2, boxdetection.x2) - max(self.x1, boxdetection.x1))
            inter_h = max(0, min(self.y2, boxdetection.y2) - max(self.y1, boxdetection.y1))
            res = inter_w * inter_h / surface
        else:
            res = 0.0
        return int(100 * res)


class ZoneDetection:
    def __init__(self):
        """ playing zone """
        # Définir les 4 points du trapèze zone de jeux (x, y)
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
        self.camera_fps = []
        self.camera_frame = None
        self.camera_frame_previews = None
        self.frame = None

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

        self.yolo_model_name = "yolo12l.pt"
        self.yolo_conf = 0.2 # seuil de confiance
        self.yolo_filter_class = [0] # 0: personne, list of class to track
        self.yolo_model = None
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.boosttrack = None
        self.bytetrack = None
        self.zone_detection = []
        self.box_detection = []
        self.tracking_detection = []
        self.tracking_seuil = 0.5 # surface minimum to link a lost box detection
        self.lost_frame_max = 30
        self.tracking_index = 0
        self.last_position_max = 5
        self.sending_url = 'http://localhost:8000/camera/detection'

        self.time_start = None
        self.time_mean = []
        self.time_fps = ''

        self.key_plot = 0
        self.key_action = ''

        self.lock = threading.Lock()
        self.running = False
        self.thread = None


    def update(self):
        while self.running:
            if self.camera is not None:
                ret, frame = self.camera.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.camera_fps.append(time.time())
                        if len(self.camera_fps) > 10:
                            del(self.camera_fps[0])
            time.sleep(0.001)

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
        cap.release()
        self.camera_usb_number = camera_usb_number

    def init_camera(self):
        """ open camera """
        if self.camera_usb_number is None:
            self.detect_camera()

        self.camera = cv2.VideoCapture(self.camera_usb_number)

        resolutions = [(800, 600), (1280, 800), (1280, 720)]
        for (w, h) in resolutions:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            rw = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (rw, rh) == (w, h):
                self.camera_width = w
                self.camera_height = h
                break

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.running = True
        self.thread.start()

    def get_camera_frame(self):

        if self.camera_frame is not None:
            self.camera_frame_previews = self.camera_frame

        if self.running:
            # input camera
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
        elif self.camera is not None:
            # input video
            ret, frame = self.camera.read()
            if not ret:
                frame = None
        else:
            frame = None

        self.camera_frame = frame
        if frame is not None and self.camera_frame_previews is None:
            self.camera_frame_previews = frame

    def stop_camera(self):
        if self.running:
            self.running = False
            self.thread.join()
        self.camera.release()


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
            fps = int(1.0 / (sum(self.time_mean) / len(self.time_mean)))
            self.time_fps = f"{int(fps)}"
        self.time_start = time_start

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

    def init_video(self, video_path=None):
        """ Ouvre un fichier vidéo pour test """

        self.camera = cv2.VideoCapture(video_path)
        # Vérifier si l'ouverture a réussi
        if not self.camera.isOpened():
            raise ValueError("Impossible d'ouvrir la source vidéo")
        self.camera_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def end_camera(self):
        """ resource free """
        self.camera.release()

    def init_model(self):
        """ Load a yolo model and tracker """
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO(self.yolo_model_name)
        self.yolo_model.to(device)
        # Initialize tracker
        self.boosttrack = BoostTrack(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),  # chemin vers ton modèle ReID
            device=device,
            half=torch.cuda.is_available()  # utilise half precision si tu veux (True pour GPU)
        )
        self.bytetrack = ByteTrack(
            match_thresh=0.9, # plus eleve plus permissif
            track_thresh=0.25,
            track_buffer=30,

            frame_rate=10,  # adapte selon ta vidéo
        )

    def get_yolo_tracking(self):
        """ return yolo tracking """
        box_detection = []

        frame = self.camera_frame.copy()

        if frame is not None:

            results = self.yolo_model.track(frame,
                                            tracker="custom_tracker.yaml",
                                            classes=self.yolo_filter_class,
                                            conf=self.yolo_conf,
                                            #imgsz=1280,
                                            persist=True)
            detections = results[0].boxes.xyxy  # Coordonnées [x1, y1, x2, y2]
            classes = results[0].boxes.cls  # Indices des classes détectées
            confs = results[0].boxes.conf  # Niveaux de confiance
            track_ids = results[0].boxes.id

            # sauvegarder les boîtes detectées
            for i, box in enumerate(detections):

                class_id = int(classes[i])
                x1, y1, x2, y2 = map(int, box)  # Conversion en int
                conf = float(confs[i])

                if track_ids is not None:
                    track_id = int(track_ids[i])
                else:
                    track_id = 0

                label = f"{track_id} - {class_id} {conf:.2f}"
                new_track = BoxDetection(x1, y1, x2, y2, label, conf, class_id, track_id)
                box_detection.append(new_track)

        self.box_detection = box_detection

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

    def show_tracking(self, frame):
        """ Add yolo box detection on frame """
        label_fps = f'FPS: {self.time_fps}'
        cv2.putText(frame, label_fps, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for bd in self.tracking_detection:
            label = f'{bd.tracking_id} - {bd.track_id} - {bd.track_boost_id} - {bd.track_byte_id} - {bd.state}'
            color = green_color
            if bd.state == 'lost':
                label += f': {bd.lost_frame}'
                color = blue_color
            if bd.state == 'new':
                color = red_color

            cv2.rectangle(frame, (bd.x1, bd.y1), (bd.x2, bd.y2), color, 2)
            cv2.putText(frame, label, (bd.x1, bd.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            xp = int(0.5 * (bd.x2 - bd.x1)) + bd.x1
            yp = bd.y2
            cv2.circle(frame, (xp, yp), 10, red_color, 2)

            for position in bd.show_last_position:
                cv2.circle(frame, position, 3, red_color, 2)

            bd.show_last_position.append((xp, yp))
            if len(bd.show_last_position) > self.last_position_max:
                del(bd.show_last_position[0])

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

    def get_box_detection_xy(self):
        """ compute xy in zone detection """
        box_detections = self.box_detection.copy()
        result = []

        for box_detection in box_detections:
            box_result = []
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
                box_result.append((x, y))

            box_detection.zone_xy = box_result
            result.append(box_detection)

        return result

    def update_xy_tracking(self):
        """ update the x and y value of box tracking """
        for box_tracking in self.tracking_detection:
            box_result = []
            xp = float((0.5 * (box_tracking.x2 - box_tracking.x1)) + box_tracking.x1)
            yp = float(box_tracking.y2)

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
                box_result.append((x, y))

            box_tracking.zone_xy = box_result
            if box_tracking.zone_xy:
                box_tracking.x = box_tracking.zone_xy[0][0]
                box_tracking.y = box_tracking.zone_xy[0][1]

            box_tracking.last_position.append((box_tracking.x, box_tracking.y))
            if len(box_tracking.last_position) > self.last_position_max:
                del(box_tracking.last_position[0])

    def update_tracking_detection_occluded(self):
        """ Update occluded box tracking """
        result = {}
        for tracking_detection in self.tracking_detection:
            tracking_detection.occluded_by_box = []
            result[tracking_detection.tracking_id] = []

        boxes_sorted = sorted(self.tracking_detection, key=lambda b: b.y2, reverse=True)
        for i, boxA in enumerate(boxes_sorted):
            for j in range(i + 1, len(boxes_sorted)):
                boxB = boxes_sorted[j]  # boxB est plus éloignée (plus haute dans l'image)
                occ = boxA.intersection_aera(boxB)
                if occ:
                    result[boxB.tracking_id].append((boxA.x1,boxA.y1,boxA.x2,boxA.y2))

        for tracking_detection in self.tracking_detection:
            tracking_detection.occluded_by_box = result[tracking_detection.tracking_id]


    def get_new_tracking_index(self):
        """ return next tracking index """
        self.tracking_index += 1
        return int(self.tracking_index)

    def compute_tracker(self):
        """ add tracking boottrack """
        # Extraire les détections
        detections = []
        tracking_index = 0
        tracking_map = {}

        for box_detection in self.box_detection:
            detections.append([
                box_detection.x1,
                box_detection.y1,
                box_detection.x2,
                box_detection.y2,
                box_detection.conf,
                box_detection.class_id])
            tracking_map[tracking_index] = box_detection
            tracking_index += 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

        if self.camera_frame is not None:

            tracked_objects = self.boosttrack.update(detections, self.camera_frame)
            for tracked_object in tracked_objects:
                tracking_index = int(tracked_object[7])
                tracking_map[tracking_index].track_boost_id = int(tracked_object[4])

            tracked_objects = self.bytetrack.update(detections, self.camera_frame)
            for tracked_object in tracked_objects:
                tracking_index = int(tracked_object[7])
                tracking_map[tracking_index].track_byte_id = int(tracked_object[4])

    def tracking_detection_save(self):
        """ init tracker index """
        for tracking_detection in self.tracking_detection:

            if tracking_detection.track_id and tracking_detection.track_id not in tracking_detection.track_ids:
                tracking_detection.track_ids.append(tracking_detection.track_id)
            tracking_detection.track_id = 0

            if tracking_detection.track_boost_id and tracking_detection.track_boost_id not in tracking_detection.track_boost_ids:
                tracking_detection.track_boost_ids.append(tracking_detection.track_boost_id)
            tracking_detection.track_boost_id = 0

            if tracking_detection.track_byte_id and tracking_detection.track_byte_id not in tracking_detection.track_byte_ids:
                tracking_detection.track_byte_ids.append(tracking_detection.track_byte_id)
            tracking_detection.track_byte_id = 0

            tracking_detection.state = 'tracking'

    def get_history_track_ids(self):
        """ return the history of track_ids """
        res = {}
        for tracking_detection in self.tracking_detection:
            for tracking_history in tracking_detection.track_ids:
                if tracking_history not in list(res.keys()):
                    res[tracking_history] = tracking_detection
        return res

    def get_history_track_boost_ids(self):
        """ return the history of track_ids """
        res = {}
        for tracking_detection in self.tracking_detection:
            for tracking_history in tracking_detection.track_boost_ids:
                if tracking_history not in list(res.keys()):
                    res[tracking_history] = tracking_detection
        return res

    def get_history_track_byte_ids(self):
        """ return the history of track_ids """
        res = {}
        for tracking_detection in self.tracking_detection:
            for tracking_history in tracking_detection.track_byte_ids:
                if tracking_history not in list(res.keys()):
                    res[tracking_history] = tracking_detection
        return res

    def compute_tracking2(self):
        """ compute new tracking with box_detection """
        self.compute_tracker()

        tracking_detection_old = self.tracking_detection.copy()

        self.tracking_detection_save()
        history_track_ids = self.get_history_track_ids()
        history_track_boost_ids = self.get_history_track_boost_ids()
        history_track_byte_ids = self.get_history_track_byte_ids()

        update_tracking_detection = []
        box_detection_ok = []

        # --------- tracking update
        for box_detection in self.box_detection:

            if box_detection.track_id in list(history_track_ids.keys()):
                tracking_detection = history_track_ids[box_detection.track_id]
            elif box_detection.track_boost_id in list(history_track_boost_ids.keys()):
                tracking_detection = history_track_boost_ids[box_detection.track_boost_id]
            elif box_detection.track_byte_id in list(history_track_byte_ids.keys()):
                tracking_detection = history_track_byte_ids[box_detection.track_byte_id]
            else:
                tracking_detection = None

            if tracking_detection:
                tracking_detection.state = 'ok'
                tracking_detection.lost_frame = 0
                tracking_detection.update_by_boxdetection(box_detection)

                box_detection_ok.append(box_detection)

        # --------- tracking new
        for box_detection in self.box_detection:
            # Check if some lost tracking_detection is corresponding
            if box_detection in box_detection_ok:
                continue
            elif box_detection.track_id or box_detection.track_byte_id or box_detection.track_boost_id:
                # New
                tracking_detection = TrackingDetection()
                tracking_detection.tracking_id = self.get_new_tracking_index()
                tracking_detection.state = 'new'
                tracking_detection.update_by_boxdetection(box_detection)
                self.tracking_detection.append(tracking_detection)



    def compute_tracking(self):
        """ compute new tracking with box_detection """
        #self.update_tracking_detection_occluded()

        tracking_detection_old = self.tracking_detection.copy()
        tracking_detection_index = {}

        old_track_ids = set(x.track_id for x in tracking_detection_old)
        box_track_ids = set(x.track_id for x in self.box_detection)

        lost_track_ids = list(old_track_ids - box_track_ids)
        new_track_ids = list(box_track_ids - old_track_ids)
        update_tracking_detection = []

        # -------- tracking lost
        detection_index = 0
        for tracking_detection in tracking_detection_old:
            if tracking_detection.track_id in lost_track_ids:
                tracking_detection.lost_frame += 1
                tracking_detection.state = 'lost'
                update_tracking_detection.append(tracking_detection)

            tracking_detection_index[tracking_detection.track_id] = detection_index
            detection_index += 1

        # --------- tracking update
        for box_detection in self.box_detection:
            if box_detection.track_id not in new_track_ids:
                tracking_detection = tracking_detection_old[tracking_detection_index[box_detection.track_id]]
                tracking_detection.state = 'ok'
                tracking_detection.lost_frame = 0
                tracking_detection.update_by_boxdetection(box_detection)
                update_tracking_detection.append(tracking_detection)

        # -------- tracking lost + opencv tracker in multiprocessing
   
        multi_tracking = []
        multi_response = []
        multi_index = 0

        for tracking_detection in tracking_detection_old:
            if tracking_detection.state == 'lost':
                queue = multiprocessing.Queue()
                process = multiprocessing.Process(target=tracking_detection.tread_opencvtracking,
                                        args=(self.camera_frame, self.camera_frame_previews, queue),
                                        name=f'{tracking_detection.tracking_id}')
                multi_response.append(queue)
                multi_tracking.append(process)
                multi_index += 1

        for tracking in multi_tracking:
            tracking.start()
        time.sleep(0.001)
        for tracking in multi_tracking:
            tracking.join()

        multi_index = 0
        for tracking_detection in tracking_detection_old:
            if tracking_detection.state == 'lost':

                (x1, y1, x2, y2) = multi_response[multi_index].get()
                tracking_detection.x1 = x1
                tracking_detection.y1 = y1
                tracking_detection.x2 = x2
                tracking_detection.y2 = y2
                multi_index += 1

        # --------- tracking new
        for box_detection in self.box_detection:
            # Check if some lost tracking_detection is corresponding
            if box_detection.track_id and box_detection.track_id in new_track_ids:
                # Check old tracking history of TrackingDetection:
                old_tracking = False
                for tracking_detection in self.tracking_detection:
                    if box_detection.track_id in tracking_detection.old_track_ids:
                        tracking_detection.update_by_boxdetection(box_detection)
                        update_tracking_detection.append(tracking_detection)
                        old_tracking = True
                        break

                if not old_tracking:
                    tracking_detection = TrackingDetection()
                    tracking_detection.tracking_id = self.get_new_tracking_index()
                    tracking_detection.state = 'new'
                    tracking_detection.update_by_boxdetection(box_detection)

                update_tracking_detection.append(tracking_detection)


        self.tracking_detection = update_tracking_detection

        # --- Update position by tracker
        self.compute_tracking_bytetrack(tracking_detection_old)
        #self.compute_tracking_boottrack()


        # --- Switch new if lost position is near
        for new_tracking_detection in self.tracking_detection:
            if new_tracking_detection.state == 'new' and new_tracking_detection.track_id:
                score_proximity = {}
                for lost_tracking_detection in self.tracking_detection:
                    if lost_tracking_detection.state == 'lost':
                        score = new_tracking_detection.intersection_boxdetection(lost_tracking_detection)
                        score_proximity[score] = lost_tracking_detection
                if score_proximity:
                    score_max = max(list(score_proximity.keys()))
                    print(score_proximity.keys())
                    if score_max >= self.tracking_seuil:
                        new_tracking_detection.tracking_id = score_proximity[score_max].tracking_id
                        score_proximity[score_max].state = 'cancel'

        # Filter the lost_frame_max detection
        finale_tracking_detection = []
        finale_tracking_id = []
        for state in ['ok', 'new', 'lost']:
            for tracking_detection in self.tracking_detection:
                if tracking_detection.state == state and tracking_detection.track_id and \
                        tracking_detection.tracking_id not in finale_tracking_id and \
                        tracking_detection.lost_frame <= self.lost_frame_max:
                    finale_tracking_detection.append(tracking_detection)
                    finale_tracking_id.append(finale_tracking_id)

        self.tracking_detection = finale_tracking_detection

        self.update_xy_tracking()

    def compute_tracking_boottrack(self):
        """ add tracking boottrack """
        # Extraire les détections
        detections = []
        tracking_index = 0
        tracking_map = {}

        for tracking_detection in self.tracking_detection:
            detections.append([
                tracking_detection.x1,
                tracking_detection.y1,
                tracking_detection.x2,
                tracking_detection.y2,
                tracking_detection.conf,
                tracking_detection.class_id])
            tracking_map[tracking_index] = tracking_detection
            tracking_index += 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

        if self.camera_frame is not None:
            tracked_objects = self.boosttrack.update(detections, self.camera_frame)

            for tracked_object in tracked_objects:
                tracking_index = tracked_object[7]
                tracking_map[tracking_index].track_boost_id = int(tracked_object[4])
                if tracking_map[tracking_index].state == 'lost':
                    tracking_map[tracking_index].x1 = int(tracked_object[0])
                    tracking_map[tracking_index].y1 = int(tracked_object[1])
                    tracking_map[tracking_index].x2 = int(tracked_object[2])
                    tracking_map[tracking_index].y2 = int(tracked_object[3])

            if tracking_map[tracking_index].state == 'new':
                print('track_boost_id---------------------------------------------------------\n', int(tracked_object[4]), '\n',
                      tracking_map[tracking_index].tracking_id, '-', tracking_map[tracking_index].track_id, '-',
                      tracking_map[tracking_index].track_boost_id, '-', tracking_map[tracking_index].track_byte_id)

    def compute_tracking_bytetrack(self, tracking_detection_old):
        """ add tracking boottrack """
        # Extraire les détections
        detections = []
        tracking_index = 0
        tracking_map = {}

        for tracking_detection in self.tracking_detection:
            detections.append([
                tracking_detection.x1,
                tracking_detection.y1,
                tracking_detection.x2,
                tracking_detection.y2,
                tracking_detection.conf,
                tracking_detection.class_id])
            tracking_map[tracking_index] = tracking_detection
            tracking_index += 1

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 6))

        if self.camera_frame is not None:
            tracked_objects = self.bytetrack.update(detections, self.camera_frame)

            for tracked_object in tracked_objects:
                tracking_index = tracked_object[7]
                tracking_map[tracking_index].track_byte_id = int(tracked_object[4])
                if tracking_map[tracking_index].state == 'lost':
                    tracking_map[tracking_index].x1 = int(tracked_object[0])
                    tracking_map[tracking_index].y1 = int(tracked_object[1])
                    tracking_map[tracking_index].x2 = int(tracked_object[2])
                    tracking_map[tracking_index].y2 = int(tracked_object[3])

                print(tracking_map[tracking_index].tracking_id, '-', tracking_map[tracking_index].track_id, '-',
                      tracking_map[tracking_index].track_boost_id, '-', tracking_map[tracking_index].track_byte_id)

    def send_tracking_datas(self):
        """ send tracking data """
        tracking_fps = self.time_mean and float(1.0 / (sum(self.time_mean) / len(self.time_mean))) or 0.0

        tracking_datas = []

        for tracking_detection in self.tracking_detection:
            if tracking_detection.state in ['new', 'ok', 'lost']:
                tracking_datas.append({
                    "tracking_id": int(tracking_detection.tracking_id),
                    "related_client_id": tracking_detection.related_client_id,
                    "posX": int(100.0 * tracking_detection.x),
                    "posY": int(100.0 * tracking_detection.y),
                    "state": tracking_detection.state,
                    "lost_frame": tracking_detection.lost_frame,
                    "zone": "game",
                    })

        data = {
            'tracking_fps': tracking_fps,
            'tracking_datas': tracking_datas,
            }

        try:
            t = threading.Thread(target=httpx.post(self.sending_url, json=data, timeout=1.0), args=(self.sending_url, data))
            t.start()
            # response = httpx.post(self.sending_url, json=data, timeout=1.0)
        except Exception as e:
            print("❌ Erreur lors de l'envoi :", e)


    def main(self):
        """ launch captation """
        parser = argparse.ArgumentParser(description="Client HTTP en threading")
        parser.add_argument("--video", required=False, help="Path of the video training")
        parser.add_argument("--show", required=False, help="View camera screen")
        parser.add_argument("--output", required=False, help="Path to save video")

        args = parser.parse_args()

        if args.video:
            self.init_video(args.video)
        else:
            detect.init_camera()

        self.init_model()
        self.load_from_json()

        out = None
        if args.output:
            # Définir le codec et créer l'objet VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.output, fourcc, 5.0, (self.camera_width, self.camera_height))

        sleep_key = False

        while True:
            start_time = time.time()
            # Capture une frame
            self.track_fps()

            self.get_camera_frame()
            camera_time = time.time()

            self.get_yolo_tracking()
            yolo_time = time.time()

            self.compute_tracking2()
            detect_time = time.time()

            if out is not None:
                out.write(self.camera_frame)

            if args.show:
                # Print information
                print('------camera_time------', camera_time - start_time)
                print('------yolo_time------', yolo_time - camera_time)
                print('------detect_time------', detect_time - yolo_time)

                # Dessiner les boîtes sur l'image originale
                frame = self.camera_frame
                frame = self.show_tracking(frame)
                frame = self.show_zone_detection(frame)

                if frame is not None:
                    cv2.imshow("Camera USB", frame)

            # Envoie les données
            self.send_tracking_datas()

            #out.write(frame)

            # Quitter avec la touche 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save_to_json()
                break
            elif key == ord('a'):
                sleep_key = True
            elif key:
                self.key_press(key)

            while True and key != ord('a'):
                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    sleep_key = False

        # Libère les ressources
        if out is not None:
            out.release()
        self.stop_camera()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detect = CameraDetection()
    detect.main()