import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from supervision import ByteTrack, Detections
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

class PlayerDetector:
    """Handles YOLO-based player detection"""
    
    def __init__(self, model_path, conf_threshold=0.4):
        self.yolo = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.yolo.names
    
    def detect(self, frame, device="cuda:0"):
        """Detect players in a frame"""
        result = self.yolo(frame, device=device)[0]
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            dets = Detections(
                xyxy=boxes.xyxy.cpu().numpy(),
                confidence=boxes.conf.cpu().numpy(),
                class_id=boxes.cls.cpu().numpy().astype(int)
            )
            return dets[dets.confidence > self.conf_threshold]
        return Detections.empty()

class PlayerTracker:
    """Handles player tracking using ByteTrack"""
    
    def __init__(self):
        self.byte_tracker = ByteTrack()
    
    def track(self, detections):
        """Track players across frames"""
        return self.byte_tracker.update_with_detections(detections)

class PlayerReIdentifier:
    """Handles player re-identification using OSNet"""
    
    def __init__(self, reid_model_path, sim_threshold=0.6):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=reid_model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.sim_threshold = sim_threshold
        self.player_features_db = {}  # track_id -> 512-d numpy feature
    
    def reidentify(self, frame, tracks, class_names):
        """Re-identify players in tracks"""
        used_ids = set()
        reid_results = []
        
        for i in range(len(tracks)):
            track_id = int(tracks.tracker_id[i])
            cls_id = int(tracks.class_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            conf = float(tracks.confidence[i])
            
            reid_id = track_id
            
            if class_names[cls_id] == 'player':
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    reid_id = self._process_player_crop(crop, track_id, used_ids)
            
            used_ids.add(reid_id)
            reid_results.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'cls': cls_id,
                'id': reid_id
            })
        
        return reid_results
    
    def _process_player_crop(self, crop, track_id, used_ids):
        """Process player crop for re-identification"""
        # Resize and save temp
        crop_resized = cv2.resize(crop, (128, 256))
        cv2.imwrite('temp.jpg', crop_resized)
        
        # Extract feature
        feat = self.extractor('temp.jpg')[0]
        if torch.is_tensor(feat):
            feat = feat.cpu().numpy()
        
        # Match against database
        if self.player_features_db:
            feats = np.stack(list(self.player_features_db.values()), axis=0)
            ids = list(self.player_features_db.keys())
            sims = cosine_similarity([feat], feats)[0]
            
            for idx in np.argsort(sims)[::-1]:
                if sims[idx] < self.sim_threshold:
                    break
                cand = ids[idx]
                if cand not in used_ids:
                    return cand
            
            # No match found, store new feature
            self.player_features_db[track_id] = feat
        else:
            # First player
            self.player_features_db[track_id] = feat
        
        return track_id

class VideoAnnotator:
    """Handles video annotation and drawing"""
    
    def __init__(self):
        self.class_colors = {
            "player": (0, 0, 0),
            "referee": (255, 0, 0),
            "ball": (0, 0, 255)
        }
    
    def draw_detections(self, frame, detections, class_names, show_tracking=True):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            conf = det['conf']
            cls_id = det['cls']
            rid = det.get('id')
            
            label = f"{class_names[cls_id]} {conf:.2f}"
            if show_tracking and rid is not None:
                label += f" ID:{rid}"
            
            color = self.class_colors.get(class_names[cls_id], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), color, 1)
            cv2.putText(frame, label, (x1+3, y1-3), font, 0.5, color, 1)
        
        return frame

class FootballPlayerReID:
    """Main class for football player re-identification pipeline"""
    
    def __init__(self, model_path, reid_model_path, conf_threshold=0.4, sim_threshold=0.6):
        self.detector = PlayerDetector(model_path, conf_threshold)
        self.tracker = PlayerTracker()
        self.reidentifier = PlayerReIdentifier(reid_model_path, sim_threshold)
        self.annotator = VideoAnnotator()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def process_video(self, video_path, output_path):
        """Process video and generate annotated output"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection
            detections = self.detector.detect(frame, self.device)
            
            # Tracking
            tracks = self.tracker.track(detections)
            
            # Re-identification
            reid_results = self.reidentifier.reidentify(frame, tracks, self.detector.class_names)
            
            # Draw and write
            annotated_frame = self.annotator.draw_detections(frame, reid_results, self.detector.class_names)
            out.write(annotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"✅ Done: wrote {frame_count} frames to {output_path}")

# === USAGE INSTRUCTIONS ===

"""
# HOW TO USE THE FOOTBALL PLAYER RE-IDENTIFICATION SYSTEM
# ========================================================

# 1. BASIC USAGE - Complete Pipeline
# ----------------------------------
# Initialize the complete pipeline with all components
reid_pipeline = FootballPlayerReID(
    model_path="Input/best.pt",                    # YOLO detection model
    reid_model_path="MyOutput/ReidModel/model.pth.tar-1",  # ReID model
    conf_threshold=0.4,                            # Detection confidence threshold
    sim_threshold=0.6                              # ReID similarity threshold
)

# Process a video file
reid_pipeline.process_video(
    video_path="Input/15sec_input_720p.mp4",       # Input video
    output_path="Output/OutputVideo/output.mp4"    # Output video
)

# 2. INDIVIDUAL COMPONENT USAGE
# -----------------------------
# Use components separately for more control

# Detection only
detector = PlayerDetector("Input/best.pt", conf_threshold=0.4)
detections = detector.detect(frame, device="cuda:0")

# Tracking only
tracker = PlayerTracker()
tracks = tracker.track(detections)

# Re-identification only
reidentifier = PlayerReIdentifier("MyOutput/ReidModel/model.pth.tar-1", sim_threshold=0.6)
reid_results = reidentifier.reidentify(frame, tracks, detector.class_names)

# Annotation only
annotator = VideoAnnotator()
annotated_frame = annotator.draw_detections(frame, reid_results, detector.class_names)

# 3. CUSTOMIZATION OPTIONS
# ------------------------
# Adjust detection confidence threshold
detector = PlayerDetector("Input/best.pt", conf_threshold=0.5)  # Higher confidence

# Adjust re-identification similarity threshold
reidentifier = PlayerReIdentifier("MyOutput/ReidModel/model.pth.tar-1", sim_threshold=0.7)  # Stricter matching

# Custom colors for annotation
annotator = VideoAnnotator()
annotator.class_colors["player"] = (0, 255, 0)  # Green for players

# 4. PROCESSING SINGLE FRAME
# --------------------------
def process_single_frame(frame, pipeline):
    detections = pipeline.detector.detect(frame, pipeline.device)
    tracks = pipeline.tracker.track(detections)
    reid_results = pipeline.reidentifier.reidentify(frame, tracks, pipeline.detector.class_names)
    return pipeline.annotator.draw_detections(frame, reid_results, pipeline.detector.class_names)

# 5. SAVE AND LOAD FEATURE DATABASE
# ---------------------------------
# The PlayerReIdentifier maintains a feature database in memory
# You can extend it to save/load features for persistent re-identification

# 6. PERFORMANCE TIPS
# -------------------
# - Use GPU for faster processing: device="cuda:0"
# - Lower conf_threshold for more detections (but more false positives)
# - Higher sim_threshold for stricter re-identification matching
# - The system automatically handles device selection (CUDA/CPU)

# 7. FILE STRUCTURE EXPECTED
# --------------------------
# Input/
#   ├── best.pt                    # YOLO detection model
#   └── 15sec_input_720p.mp4      # Input video
# MyOutput/
#   └── ReidModel/
#       └── model.pth.tar-1       # ReID model
# Output/
#   └── OutputVideo/              # Output directory

# 8. DEPENDENCIES REQUIRED
# ------------------------
# - ultralytics (for YOLO)
# - supervision (for ByteTrack)
# - torchreid (for ReID models)
# - opencv-python (for video processing)
# - torch (for deep learning)
# - numpy (for numerical operations)
# - scikit-learn (for cosine similarity)
"""
