import cv2
import numpy as np
import torch
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity



def draw_detections(frame, detections, class_names, show_tracking=True):
  # === Drawing Utilities ===
    CLASS_COLORS = {
        "player": (0, 0, 0),
        "referee": (255, 0, 0),
        "ball": (0, 0, 255)
    }
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['conf']
        cls_id = det['cls']
        rid = det.get('id')
        label = f"{class_names[cls_id]} {conf:.2f}"
        if show_tracking and rid is not None:
            label += f" ID:{rid}"
        color = CLASS_COLORS.get(class_names[cls_id], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, 1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3), font, 0.5, color, 1)
    return frame

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def load_models(model_path, reid_path, device):
    yolo = YOLO(model_path)
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=reid_path,
        device=device
    )
    byte_tracker = ByteTrack()
    player_features_db = {}
    return yolo, extractor, byte_tracker, player_features_db

def setup_video_io(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    return cap, out, fps, w, h

def get_detections(frame, yolo, device, conf_thresh=0.4):
    result = yolo(frame, device=device)[0]
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        dets = Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int)
        )
        return dets[dets.confidence > conf_thresh]
    return Detections.empty()

def process_tracks(frame, tracks, class_names, extractor, player_features_db, sim_threshold=0.6):
    used_ids = set()
    draw_list = []
    for i in range(len(tracks)):
        track_id = int(tracks.tracker_id[i])
        cls_id = int(tracks.class_id[i])
        x1, y1, x2, y2 = map(int, tracks.xyxy[i])
        conf = float(tracks.confidence[i])
        reid_id = track_id

        if class_names[cls_id] == 'player':
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (128, 256))
            feat = extractor(crop_resized)[0]
            feat = feat.cpu().numpy() if torch.is_tensor(feat) else feat

            if player_features_db:
                feats = np.stack(list(player_features_db.values()), axis=0)
                ids = list(player_features_db.keys())
                sims = cosine_similarity([feat], feats)[0]
                for idx in np.argsort(sims)[::-1]:
                    if sims[idx] < sim_threshold:
                        break
                    cand = ids[idx]
                    if cand not in used_ids:
                        reid_id = cand
                        break
                else:
                    player_features_db[track_id] = feat
            else:
                player_features_db[track_id] = feat

        used_ids.add(reid_id)
        draw_list.append({
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'cls': cls_id,
            'id': reid_id
        })

    return draw_list
