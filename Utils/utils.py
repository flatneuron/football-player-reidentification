import cv2
import numpy as np
import torch
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

# Draws bounding boxes and labels for detections on a frame
def draw_detections(frame, detections, class_names, show_tracking=True):
    """
    Draw bounding boxes and class labels (and optionally tracking IDs) on the frame.
    """
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

# Returns the device string for torch (cuda if available, else cpu)
def get_device():
    """Return the best available device for torch (cuda or cpu)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"

# Loads YOLO, feature extractor, and tracker models
def load_models(model_path, reid_path, device):
    """
    Load YOLO detection model, re-identification feature extractor, and ByteTrack tracker.
    """
    yolo = YOLO(model_path)
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=reid_path,
        device=device
    )
    byte_tracker = ByteTrack()
    player_features_db = {}  # Stores features for each player for re-identification
    return yolo, extractor, byte_tracker, player_features_db

# Sets up video capture and output writer
def setup_video_io(video_path, output_path):
    """
    Initialize video capture and output writer for processing video frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    return cap, out, fps, w, h

# Runs YOLO detection and returns filtered detections
def get_detections(frame, yolo, device, conf_thresh=0.4):
    """
    Run YOLO model on a frame and return detections above confidence threshold.
    """
    result = yolo(frame, device=device)[0]
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        dets = Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int)
        )
        return dets[dets.confidence > conf_thresh]  # Filter by confidence
    return Detections.empty()

# Processes tracks, extracts features, and assigns consistent IDs
def process_tracks(frame, tracks, class_names, extractor, player_features_db, sim_threshold=0.6):
    """
    For each track, extract features for players and assign consistent re-identification IDs using cosine similarity.
    """
    used_ids = set()
    draw_list = []
    for i in range(len(tracks)):
        track_id = int(tracks.tracker_id[i])
        cls_id = int(tracks.class_id[i])
        x1, y1, x2, y2 = map(int, tracks.xyxy[i])
        conf = float(tracks.confidence[i])
        reid_id = track_id  # Default to tracker ID

        if class_names[cls_id] == 'player':
            crop = frame[y1:y2, x1:x2]  # Crop player region from frame
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (128, 256))  # Resize for feature extractor
            feat = extractor(crop_resized)[0]  # Extract feature vector
            feat = feat.cpu().numpy() if torch.is_tensor(feat) else feat

            if player_features_db:
                feats = np.stack(list(player_features_db.values()), axis=0)  # All stored features
                ids = list(player_features_db.keys())
                sims = cosine_similarity([feat], feats)[0]  # Compute similarity to all stored features
                for idx in np.argsort(sims)[::-1]:  # Sort by similarity descending
                    if sims[idx] < sim_threshold:
                        break  # Stop if similarity below threshold
                    cand = ids[idx]
                    if cand not in used_ids:
                        reid_id = cand  # Assign existing ID if not used in this frame
                        break
                else:
                    player_features_db[track_id] = feat  # New player, add to DB
            else:
                player_features_db[track_id] = feat  # First player, add to DB

        used_ids.add(reid_id)
        draw_list.append({
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'cls': cls_id,
            'id': reid_id
        })

    return draw_list
