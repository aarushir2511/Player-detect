import torch
import numpy as np
from torchvision import transforms
import cv2
import torchreid

from .tracker import Tracker
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from . import preprocessing

class DeepSort:
    def __init__(self, max_dist=0.3, max_iou_distance=0.7, max_age=50, n_init=3):
        """
        Improved DeepSORT with better feature extraction and error handling
        """
        print("🔄 Initializing DeepSORT...")
        
        # ✅ Try to load appearance re-ID model with fallback
        try:
            self.extractor = torchreid.models.build_model(
                name='osnet_ain_x1_0',
                num_classes=1000,
                pretrained=True
            )
            self.extractor.eval()
            self.use_features = True
            print("✅ Loaded OSNet feature extractor")
        except Exception as e:
            print(f"⚠️  Failed to load OSNet: {e}")
            print("🔄 Falling back to basic color features...")
            self.extractor = None
            self.use_features = False

        # ✅ Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.extractor is not None:
            self.extractor.to(self.device)
        
        print(f"🔧 Using device: {self.device}")

        # ✅ Image preprocessing for OSNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ✅ Deep SORT tracker with more lenient parameters
        metric = NearestNeighborDistanceMetric(
            metric="cosine",
            matching_threshold=max_dist,
            budget=100
        )
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )
        
        print("✅ DeepSORT initialized successfully")

    def extract_color_histogram(self, crop):
        """
        Fallback feature extraction using color histogram
        """
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize and concatenate
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
            
            feature = np.concatenate([h_hist, s_hist, v_hist])
            
            # Pad to 512 dimensions if needed
            if len(feature) < 512:
                feature = np.pad(feature, (0, 512 - len(feature)), 'constant')
            else:
                feature = feature[:512]
            
            return feature
        except Exception as e:
            print(f"❌ Color histogram extraction failed: {e}")
            return np.random.rand(512) * 0.1  # Very small random features

    def update_tracks(self, bbox_xywh_conf, frame):
        """
        Update tracking with improved error handling
        """
        if len(bbox_xywh_conf) == 0:
            # Run tracker prediction step even with no detections
            self.tracker.predict()
            self.tracker.update([])
            return []
        
        # bbox format: [x, y, w, h, conf]
        bbox_xywh = np.array([b[:4] for b in bbox_xywh_conf])
        confidences = np.array([b[4] for b in bbox_xywh_conf])

        features = []
        
        for i, box in enumerate(bbox_xywh):
            x, y, w, h = box.astype(int)
            
            # ✅ Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            # ✅ Extract crop with padding
            pad = 5
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame_w)
            y2 = min(y + h + pad, frame_h)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                print(f"⚠️  Invalid crop for detection {i}: {crop.shape}")
                features.append(np.random.rand(512) * 0.1)
                continue

            try:
                if self.use_features and self.extractor is not None:
                    # Use deep features
                    crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.extractor(crop_tensor).cpu().numpy()[0]
                else:
                    # Use color histogram features
                    feat = self.extract_color_histogram(crop)

                # ✅ Normalize feature
                norm = np.linalg.norm(feat)
                if norm < 1e-6:
                    print(f"⚠️  Very small feature norm: {norm}")
                    feat = np.random.rand(512) * 0.1
                else:
                    feat = feat / norm

                features.append(feat)

            except Exception as e:
                print(f"❌ Feature extraction failed for detection {i}: {e}")
                features.append(np.random.rand(512) * 0.1)

        features = np.array(features)

        # ✅ Create detections
        detections = [
            Detection(bbox, conf, feat)
            for bbox, conf, feat in zip(bbox_xywh, confidences, features)
        ]

        # ✅ Tracking step
        try:
            self.tracker.predict()
            self.tracker.update(detections)
        except Exception as e:
            print(f"❌ Tracker update failed: {e}")
            return []

        # ✅ Return (track, feature) pairs for all confirmed tracks
        output = []
        for track in self.tracker.tracks:
            if track.is_confirmed():
                # Find corresponding feature
                track_feat = None
                if hasattr(track, 'features') and len(track.features) > 0:
                    track_feat = track.features[-1]  # Use most recent feature
                else:
                    track_feat = np.random.rand(512) * 0.1
                
                output.append((track, track_feat))

        return output