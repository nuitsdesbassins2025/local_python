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
from boxmot import OcSort
from pathlib import Path
import threading
import argparse
from collections import Counter


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
        self.track_ocsort_id = 0  # ocsort tracking 0 = None



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
        self.mean_h = []
        self.mean_center = []
        self.center_pred = (0, 0)

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
        self.track_ocsort_id = 0  # ocsort tracking 0 = None

        self.track_ids = []
        self.track_boost_ids = []
        self.track_byte_ids = []
        self.track_ocsort_ids = []

        self.tracker_fields = []
        self.not_track_ids = []
        self.not_track_boost_ids = []
        self.not_track_byte_ids = []
        self.not_track_ocsort_ids = []

        self.tracking_ok = 0
        self.tracking_ko = 0

        self.tracking_id = 0
        self.related_client_id = ''
        self.lost_frame = 0
        self.zone_xy = []
        self.state = 'new'
        self.conf = 0
        self.tracker = None

    def predict_next_point(self):
        """ estimation of next point """

        if len(self.mean_center) > 3:
            points = self.mean_center
            velocities= []
            # Calcul des vitesses entre points successifs
            for i in range(3):
                velocities.append((points[-(i + 1)][0] - points[-(i + 2)][0], points[-(i + 1)][1] - points[-(i + 2)][1]))

            if velocities:
                # Moyenne des vitesses
                avg_vx = 0
                avg_vy = 0
                for v in velocities:
                    avg_vx += v[0]
                    avg_vy += v[1]

                    avg_vx = int(avg_vx / len(velocities))
                    avg_vy = int(avg_vy / len(velocities))

                # Dernier point connu
                last_x, last_y = points[-1]

                # Prédiction du prochain point
                self.center_pred = (last_x + avg_vx, last_y + avg_vy)


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

        for field_track in self.tracker_fields:
            tracker_id = getattr(self, field_track, 0)
            not_tracker_ids = getattr(self, 'not_' + field_track + 's', 0)
            tracker_boxdetection_id = getattr(boxdetection, field_track, 0)
            if tracker_id not in not_tracker_ids:
                setattr(self, field_track, tracker_boxdetection_id)

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
            # TrackerMIL_create TrackerCSRT_create
            tracker = cv2.legacy.TrackerMIL_create()
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
        self.camera_width = 800
        self.camera_height = 600
        self.camera_fps = []
        self.camera_frame = None
        self.camera_frame_previews = None
        self.frame = None

        # resolution: 1280 800 FPS: 11
        # resolution: 800 600 FPS: 20
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

        self.yolo_model_name = "yolo12x.pt"
        self.yolo_conf = 0.1 # seuil de confiance
        self.yolo_filter_class = [0] # 0: personne, list of class to track
        self.yolo_model = None
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.boosttrack = None
        self.bytetrack = None
        self.ocsort = None
        self.tracker_fields = ['track_id', 'track_boost_id', 'track_byte_id', 'track_ocsort_id']
        self.zone_detection = []
        self.box_detection = []
        self.tracking_detection = []
        self.tracking_seuil = 0.5 # surface minimum to link a lost box detection
        self.lost_frame_max = 20
        self.tracking_index = 0
        self.last_position_max = 5
        self.sending_url = 'http://localhost:8000/camera/detection'

        self.stat_ok = [0,] * len(self.tracker_fields)
        self.stat_lost = [0,] * len(self.tracker_fields)
        self.stat_new = [0,] * len(self.tracker_fields)

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
            self.camera_frame_previews = self.camera_frame.copy()

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
        """
        Initializes the BoostTrack tracker with various parameters.

        Args:
            reid_weights: Path to the re-identification model weights.
            device: Device to run the model on (e.g., 'cpu', 'cuda').
            half: Whether to use half-precision for computations.
            max_age: 60, Maximum allowed frames without update.
            min_hits: 3, Minimum hits required to output a track.
            det_thresh: 0.6, Detection confidence threshold.
            iou_threshold: 0.3, IoU threshold for association.
            use_ecc: Whether to use ECC for camera motion compensation.
            min_box_area: 10, Minimum box area for detections.
            aspect_ratio_thresh: Aspect ratio threshold for detections.
            cmc_method: Method for camera motion compensation.
            lambda_iou: 0.5 Weight for IoU-based association.
            lambda_mhd: 0.25 Weight for Mahalanobis distance-based association.
            lambda_shape: 0.25 Weight for shape-based association.
            use_dlo_boost: true Whether to use DLO boost.
            use_duo_boost: true Whether to use DUO boost.
            dlo_boost_coef: 0.65 Coefficient for DLO boost.
            s_sim_corr: Whether to use shape similarity correction.
            use_rich_s: Whether to use rich shape features.
            use_sb: false Whether to use soft-BIoU.
            use_vt: false Whether to use visual tracking.
            with_reid: Whether to use re-identification.
            per_class: If True, enables per-class tracking, where tracks are managed separately for each class.
        """
        self.boosttrack = BoostTrack(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),  # chemin vers ton modèle ReID
            det_thresh=0.5,
            min_hits=1,
            max_age=20,
            device=device,
            half=torch.cuda.is_available()  # utilise half precision si tu veux (True pour GPU)
        )
        """
        BYTETracker: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

        Args:
            min_conf (float, optional): 0.1 Threshold for detection confidence. Detections below this threshold are discarded.
            track_thresh (float, optional): 0.45 Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
            match_thresh (float, optional): 0.8 Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
            track_buffer (int, optional): 25 Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
            frame_rate (int, optional): 30 Frame rate of the video being processed. Used to scale the track buffer size.
            per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        """
        self.bytetrack = ByteTrack(
            match_thresh=0.8, # plus eleve plus permissif
            track_thresh=0.45,
            track_buffer=20,
            frame_rate=10,  # adapte selon ta vidéo
        )
        """
        OCSort Tracker: A tracking algorithm that utilizes motion-based tracking.

        Args:
            per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
            det_thresh (float, optional): Detection confidence threshold. Detections below this threshold are ignored in the first association step.
            max_age (int, optional): 30 Maximum number of frames to keep a track alive without any detections.
            min_hits (int, optional): 3 Minimum number of hits required to confirm a track.
            asso_threshold (float, optional): 0.3 Threshold for the association step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
            delta_t (int, optional): 3 Time delta for velocity estimation in Kalman Filter.
            asso_func (str, optional): Association function to use for data association. Options include "iou" for IoU-based association.
            inertia (float, optional): 0.2 Weight for inertia in motion modeling. Higher values make tracks less responsive to changes.
            use_byte (bool, optional): false Whether to use BYTE association in the second association step.
            Q_xy_scaling (float, optional): 0.01 Scaling factor for the process noise covariance in the Kalman Filter for position coordinates.
            Q_s_scaling (float, optional): 0.0001 Scaling factor for the process noise covariance in the Kalman Filter for scale coordinates.
        """
        self.ocsort = OcSort(
            det_thresh=0.25,
            max_age=20,
            inertia=0.3,
            min_hits=1,
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
        label_fps = f'{frame.shape} FPS: {self.time_fps}'
        cv2.putText(frame, label_fps, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'ok   : {self.stat_ok}', (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'lost  : {self.stat_lost}', (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'new  : {self.stat_new}', (10, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for bd in self.tracking_detection:
            label = f'{bd.tracking_id}: {bd.track_id}.{bd.track_boost_id}.{bd.track_byte_id}.{bd.track_ocsort_id}.{bd.state}'
            color = green_color
            if bd.state == 'lost':
                label += f':{bd.lost_frame}'
                color = blue_color
            if bd.state == 'new':
                color = red_color
            label += f'-{int(100.0 * bd.conf)}%'
            cv2.rectangle(frame, (bd.x1, bd.y1), (bd.x2, bd.y2), color, 2)
            cv2.putText(frame, label, (bd.x1, bd.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f' {bd.tracking_ok}', (bd.x1, bd.y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 2)
            cv2.putText(frame, f' {bd.tracking_ko}', (bd.x1, bd.y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 2)


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

            # Show mean position
            if bd.mean_center:
                cv2.circle(frame, bd.mean_center[-1], 5, (0,0,0), 2)
            if bd.center_pred:
                cv2.circle(frame, bd.center_pred, 8, (255, 255, 255), 2)

        # Show no tracking box
        for bd_void in self.box_detection:
            if not (bd_void.track_id and bd_void.track_boost_id and bd_void.track_byte_id):
                color = (0, 0, 0)
                cv2.rectangle(frame, (bd_void.x1, bd_void.y1), (bd_void.x2, bd_void.y2), (255,255,255), 1)

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

    def upadte_mean_h_w(self):
        """ Update the value of mean h and w """
        for box_tracking in self.tracking_detection:

            box_tracking.mean_h.append(box_tracking.y2 - box_tracking.y1)
            box_tracking.mean_w.append(box_tracking.x2 - box_tracking.x1)
            box_tracking.mean_center.append((
                box_tracking.x1 + int((box_tracking.x2 - box_tracking.x1) / 2),
                box_tracking.y1 + int((box_tracking.y2 - box_tracking.y1) / 2)
                ))

            for list_max in [box_tracking.mean_h, box_tracking.mean_w, box_tracking.mean_center]:
                if len(list_max) > 5:
                    del(list_max[0])

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

            tracked_objects = self.ocsort.update(detections, self.camera_frame)
            for tracked_object in tracked_objects:
                tracking_index = int(tracked_object[7])
                tracking_map[tracking_index].track_ocsort_id = int(tracked_object[4])


    def tracking_detection_save(self):
        """ init tracker index """
        max_tracking = 3
        history_track = self.get_history_track()

        for tracking_detection in self.tracking_detection:
            for field_track in self.tracker_fields:
                track_id = getattr(tracking_detection, field_track, 0)
                if not track_id:
                    continue

                for other_tracking_detection in self.tracking_detection:
                    not_track_ids = getattr(other_tracking_detection, 'not_' + field_track + 's', [])
                    if other_tracking_detection != tracking_detection:
                        if track_id not in not_track_ids:
                            not_track_ids.append(track_id)
                        if track_id in getattr(other_tracking_detection, field_track + 's'):
                            getattr(other_tracking_detection, field_track + 's').remove(track_id)

        for tracking_detection in self.tracking_detection:
            for field_track in self.tracker_fields:
                track_id = getattr(tracking_detection, field_track, 0)
                track_ids = getattr(tracking_detection, field_track + 's', [])
                not_track_ids = getattr(tracking_detection, 'not_' + field_track + 's', [])
                if not track_id:
                    continue

                if track_id not in not_track_ids and track_id not in track_ids:
                    track_ids.append(track_id)
                    if len(track_ids) > max_tracking:
                        del (track_ids[0])
                track_id = 0

            tracking_detection.state = 'tracking'

    def get_history_track(self):
        """ return a dic with """
        res = {}
        for field_track in self.tracker_fields:
            res[field_track] = {}

        for tracking_detection in self.tracking_detection:
            for field_track in self.tracker_fields:
                for tracking_history in getattr(tracking_detection, field_track + 's', []):
                    if tracking_history not in list(res[field_track].keys()):
                        res[field_track][tracking_history] = tracking_detection
        return res



    def compute_tracking2(self):
        """ compute new tracking with box_detection """

        self.compute_tracker()

        tracking_detection_old = self.tracking_detection.copy()

        self.tracking_detection_save()


        history_track = self.get_history_track()
        update_tracking_detection = []
        box_detection_ok = []

        # --------- tracking update
        box_detection_tracking = []

        for box_detection in self.box_detection:
            tracking_ids = []
            for field_track in self.tracker_fields:
                tracker_id = getattr(box_detection, field_track, 0)
                if tracker_id in list(history_track[field_track].keys()):
                    tracking_ids.append(history_track[field_track][tracker_id])

            counter = Counter(tracking_ids)
            if counter:
                most_box_tracking, count = counter.most_common(1)[0]
                for field_track in self.tracker_fields:
                    tracker_id = getattr(box_detection, field_track, 0)
                    if tracker_id in list(history_track[field_track].keys()) and history_track[field_track][tracker_id] != most_box_tracking:
                        setattr(box_detection, field_track, 0)

                most_box_tracking.state = 'ok'
                most_box_tracking.lost_frame = 0
                most_box_tracking.tracking_ok += 1
                most_box_tracking.update_by_boxdetection(box_detection)
                box_detection_ok.append(box_detection)

        # --------- tracking lost
        for tracking_detection in self.tracking_detection:
            if tracking_detection.state == 'tracking':
                tracking_detection.state = 'lost'
                tracking_detection.lost_frame += 1
                tracking_detection.tracking_ko += 1

                # --------- Check new box_detection
                score_proximity = {}
                for box_detection in self.box_detection:
                    # Check if some box_detection is corresponding
                    if box_detection in box_detection_ok:
                        continue
                    elif box_detection.track_id or box_detection.track_byte_id or box_detection.track_boost_id:
                        score = tracking_detection.intersection_boxdetection(box_detection)
                        score_proximity[score] = box_detection

                if False and score_proximity:
                    score_max = max(list(score_proximity.keys()))
                    if score_max >= self.tracking_seuil:
                        tracking_detection.state = 'ok'
                        if self.lost_frame_max < tracking_detection.lost_frame:
                            self.lost_frame_max = tracking_detection.lost_frame
                        tracking_detection.lost_frame = 0
                        tracking_detection.update_by_boxdetection(box_detection)

                        box_detection_ok.append(box_detection)

        # --------- tracking new
        for box_detection in self.box_detection:
            # Check if some lost tracking_detection is corresponding
            if box_detection in box_detection_ok:
                continue
            elif box_detection.track_id:
                # New
                tracking_detection = TrackingDetection()
                tracking_detection.tracking_id = self.get_new_tracking_index()
                tracking_detection.state = 'new'
                tracking_detection.tracker_fields = self.tracker_fields
                tracking_detection.update_by_boxdetection(box_detection)
                self.tracking_detection.append(tracking_detection)

        # Delete old box_detection
        for tracking_detection in self.tracking_detection:
            if tracking_detection.lost_frame > int(2 * self.lost_frame_max):
                self.tracking_detection.remove(tracking_detection)

        # statistic
        self.update_statistic()

        # Update XY
        self.update_tracking_detection_occluded()
        self.upadte_mean_h_w()
        for tracking_detection in self.tracking_detection:
            tracking_detection.predict_next_point()
        self.update_xy_tracking()

    def update_statistic(self):
        """ update statistic of tracker """
        for tracking_detection in self.tracking_detection:
            if tracking_detection.state == 'ok':
                statistic_ok = self.stat_ok.copy()
                statistic_lost = self.stat_lost.copy()
                for i, field_track in enumerate(self.tracker_fields):
                    if getattr(tracking_detection, field_track):
                        statistic_ok[i] += 1
                    else:
                        statistic_lost[i] += 1
                self.stat_ok = statistic_ok
                self.stat_lost = statistic_lost

            if tracking_detection.state == 'new':
                statistic_new = self.stat_new.copy()
                for i, field_track in enumerate(self.tracker_fields):
                    if getattr(tracking_detection, field_track):
                        statistic_new[i] += 1
                self.stat_new = statistic_new

            if tracking_detection.state == 'lost':
                statistic_lost = self.stat_lost.copy()
                for i, field_track in enumerate(self.tracker_fields):
                    statistic_lost[i] += 1
                self.stat_lost = statistic_lost

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
        parser.add_argument("--step", required=False, help="Step by step")

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

            while args.step and key != ord('a'):
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