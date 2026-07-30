import os
import tempfile
import cv2
import time
from pathlib import Path

import streamlit as st
import math
import numpy as np
# Import ultralytics only when needed (inside load_model) to avoid import-time
# failures in environments where torch is not yet installed. This prevents the
# Streamlit process from exiting immediately inside a Docker container while
# dependencies are being installed or when a CPU-only torch is later added.


st.set_page_config(page_title="Drowning Detection Dashboard", layout="wide")

# base dir used by multiple blocks
BASE_DIR = Path(__file__).resolve().parent

# Optional: support the DrownDetect model if available (model.pth + lb.pkl)
HAS_DROWN = False
DROWN_MODEL = None
DROWN_LB = None
DROWN_AUG = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import joblib
    import albumentations
    from PIL import Image
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    # try to locate the Drowning-Detection--master folder near the workspace
    alt_dir = None
    possible = [
        BASE_DIR.parent / "Drowning-Detection--master",
        BASE_DIR / ".." / "Drowning-Detection--master",
        Path.cwd() / "Drowning-Detection--master",
    ]
    for p in possible:
        if p.exists():
            alt_dir = p
            break

    if alt_dir is not None:
        pth = alt_dir / "model.pth"
        lbp = alt_dir / "lb.pkl"
        if pth.exists() and lbp.exists():
            # define CustomCNN locally (same as DrownDetect)
            class CustomCNN(nn.Module):
                def __init__(self):
                    super(CustomCNN, self).__init__()
                    self.conv1 = nn.Conv2d(3, 16, 5)
                    self.conv2 = nn.Conv2d(16, 32, 5)
                    self.conv3 = nn.Conv2d(32, 64, 3)
                    self.conv4 = nn.Conv2d(64, 128, 5)
                    self.fc1 = nn.Linear(128, 256)
                    # fc2 will be set after loading lb to match classes
                    self.fc2 = None
                    self.pool = nn.MaxPool2d(2, 2)

                def forward(self, x):
                    x = self.pool(F.relu(self.conv1(x)))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = self.pool(F.relu(self.conv3(x)))
                    x = self.pool(F.relu(self.conv4(x)))
                    bs, _, _, _ = x.shape
                    x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
                    x = F.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x

            try:
                DROWN_LB = joblib.load(str(lbp))
                model_dd = CustomCNN()
                # set fc2 with correct output dim
                model_dd.fc2 = nn.Linear(256, len(DROWN_LB.classes_))
                model_dd.load_state_dict(torch.load(str(pth), map_location='cpu'))
                model_dd.eval()
                DROWN_MODEL = model_dd
                DROWN_AUG = albumentations.Compose([albumentations.Resize(224, 224)])
                HAS_DROWN = True
            except Exception:
                HAS_DROWN = False

# Tapo live stream source.
TAPO_STREAM_URL = 'rtsp://AquaGuard:AQGuardCam@192.168.68.53:554/stream1'

# Light/demo mode: when STREAMLIT_LIGHT_MODE env var is set (or user chooses),
# avoid heavy imports (torch/ultralytics) and play a local sample video or image.
LIGHT_MODE = os.environ.get('STREAMLIT_LIGHT_MODE', '0') in ('1', 'true', 'True')


@st.cache_resource
def load_model(model_path: str = None):
    if LIGHT_MODE:
        # In light/demo mode, skip heavy model loading.
        st.info("Running in LIGHT MODE: model loading is disabled. App will show demo output.")
        return None

    # Import YOLO here to avoid import-time dependency issues in Docker/startup.
    try:
        from ultralytics import YOLO
    except Exception as e:
        st.error(f"Failed to import ultralytics.YOLO: {e}. Ensure 'ultralytics' is installed.")
        return None

    if model_path is None:
        # Try to use model.pt in repository root or provided path
        candidate = BASE_DIR / "model.pt"
        alt1 = BASE_DIR / "drowning_detection_master.pt"
        alt2 = BASE_DIR / "drowning-detection-master.pt"
        if candidate.exists():
            model_path = str(candidate)
        elif alt1.exists():
            model_path = str(alt1)
        elif alt2.exists():
            model_path = str(alt2)
        else:
            model_path = None
    if model_path is None:
        st.error("Model file not found. Place `model.pt` in the project root or specify a path.")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model from {model_path}: {e}")
        return None


def process_video(model, input_path, output_path, conf=0.4, frame_skip=1):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps / frame_skip if frame_skip>0 else fps, (width, height))

    frame_idx = 0
    processed = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Run detection
        results = model.predict(frame, conf=conf, verbose=False)

        # ultralytics returns list; take first
        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                    # Only draw red boxes for class 0 (drowning/swimmer of interest)
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confscore = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0
                        label = f"{cls}:{confscore:.2f}"
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        processed += 1
        frame_idx += 1

    cap.release()
    out.release()
    elapsed = time.time() - start
    return processed, elapsed


def stream_process_video(model, input_path, conf=0.4, frame_skip=1, placeholder=None):
    """Stream processed frames to the Streamlit UI in (near) real-time.
    Returns the number of frames processed and elapsed time.
    """
    # Persist capture and frame index in session_state so UI changes (reruns) don't reset playback
    if st.session_state.get('live_video_path') != str(input_path) or 'cap_live' not in st.session_state:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        st.session_state['cap_live'] = cap
        st.session_state['live_video_path'] = str(input_path)
        st.session_state['live_frame_idx'] = 0
        st.session_state['live_processed'] = 0
    else:
        cap = st.session_state.get('cap_live')
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {input_path}")
            st.session_state['cap_live'] = cap

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # initial delay; we'll recompute per-loop using live frame_skip from session_state
    delay = 1.0 / (fps / frame_skip) if (fps and frame_skip) else 0.04

    if placeholder is None:
        placeholder = st.empty()
    info = st.empty()
    stats = st.empty()

    # If we have a cached last-frame JPEG from a previous run, show it immediately
    # This avoids a brief white flash when the Streamlit script reruns.
    if st.session_state.get('latest_frame_jpeg', None) is not None:
        try:
            placeholder.image(st.session_state['latest_frame_jpeg'], width=1200)
        except Exception:
            # fallback: ignore and continue
            pass

    # Simple centroid tracker
    class CentroidTracker:
        def __init__(self, maxDisappeared=30):
            self.nextObjectID = 0
            self.objects = dict()
            self.disappeared = dict()
            self.maxDisappeared = maxDisappeared

        def register(self, centroid):
            self.objects[self.nextObjectID] = centroid
            self.disappeared[self.nextObjectID] = 0
            self.nextObjectID += 1

        def deregister(self, objectID):
            del self.objects[objectID]
            del self.disappeared[objectID]

        def update(self, rects):
            # rects: list of bounding boxes [x1,y1,x2,y2]
            if len(rects) == 0:
                for objectID in list(self.disappeared.keys()):
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                return self.objects

            inputCentroids = []
            for (x1, y1, x2, y2) in rects:
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                inputCentroids.append((cX, cY))

            if len(self.objects) == 0:
                for c in inputCentroids:
                    self.register(c)
            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())

                # compute distance matrix between objectCentroids and inputCentroids
                D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype="float")
                for i in range(len(objectCentroids)):
                    for j in range(len(inputCentroids)):
                        D[i, j] = math.hypot(objectCentroids[i][0] - inputCentroids[j][0], objectCentroids[i][1] - inputCentroids[j][1])

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()
                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    usedRows.add(row)
                    usedCols.add(col)

                # register new input centroids
                for i in range(len(inputCentroids)):
                    if i not in usedCols:
                        self.register(inputCentroids[i])

                # check disappeared
                for i in range(len(objectCentroids)):
                    if i not in usedRows:
                        objectID = objectIDs[i]
                        self.disappeared[objectID] += 1
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)

            return self.objects

    tracker = CentroidTracker(maxDisappeared=30)

    frame_idx = int(st.session_state.get('live_frame_idx', 0))
    processed = int(st.session_state.get('live_processed', 0))
    start = time.time()

    looping_file = False
    if isinstance(input_path, Path) and input_path.exists() and input_path.suffix.lower() in ['.mp4', '.mov', '.mkv']:
        looping_file = True

    while True:
        ret, frame = cap.read()
        if not ret:
            # if file and looping, restart capture
            try:
                cap.release()
            except Exception:
                pass
            if looping_file:
                cap = cv2.VideoCapture(str(input_path))
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        # Read dynamic frame_skip from sidebar via session_state so changes apply immediately
        try:
            current_skip = int(st.session_state.get('frame_skip', frame_skip))
        except Exception:
            current_skip = frame_skip

        if frame_idx % max(1, current_skip) != 0:
            frame_idx += 1
            st.session_state['live_frame_idx'] = frame_idx
            continue

        t0 = time.time()

        # Read current confidence from session state so adjustments apply immediately
        try:
            current_conf = float(st.session_state.get('confidence', conf))
        except Exception:
            current_conf = conf
        # recompute delay based on current_skip
        try:
            delay = 1.0 / (fps / max(1, current_skip)) if fps else 0.04
        except Exception:
            delay = 0.04
        # Run detection
        results = model.predict(frame, conf=current_conf, verbose=False)

        # Draw boxes if present and collect rects for tracking
        rects = []
        if len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confscore = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0

                        is_drown = False
                        # If DrownDetect model is available, use it to classify the crop
                        if HAS_DROWN and DROWN_MODEL is not None:
                            try:
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                augimg = DROWN_AUG(image=np.array(pil_image))['image']
                                arr = np.transpose(augimg, (2,0,1)).astype(np.float32)
                                tensor = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
                                with torch.no_grad():
                                    out = DROWN_MODEL(tensor)
                                    _, pred = torch.max(out.data, 1)
                                    label_name = DROWN_LB.classes_[pred]
                                    if label_name == 'drowning':
                                        is_drown = True
                            except Exception:
                                is_drown = False
                        else:
                            # fallback: only consider ultralytics class 0 as drowning
                            if cls == 0:
                                is_drown = True

                        # Decide whether to draw based on toggles
                        show_blue = st.session_state.get('show_blue', True)
                        show_red = st.session_state.get('show_red', True)
                        if is_drown and show_red:
                            color = (0, 0, 255)
                            label = f"drowning:{confscore:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            rects.append((x1, y1, x2, y2))
                        elif not is_drown and show_blue:
                            color = (255, 0, 0)
                            label = f"safe:{confscore:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            rects.append((x1, y1, x2, y2))
                    except Exception:
                        continue

        objects = tracker.update(rects)
        # Do not draw tracker IDs on the UI; only red bounding boxes are shown

        # Encode the original BGR frame as JPEG bytes for a stable UI update.
        # Encoding the BGR image avoids channel swapping issues that occur
        # when encoding an already-converted RGB array with OpenCV.
        try:
            jpeg = cv2.imencode('.jpg', frame)[1].tobytes()
            st.session_state['latest_frame_jpeg'] = jpeg
            placeholder.image(jpeg, width=1200)
        except Exception:
            # If encoding fails for any reason, fall back to converting to RGB
            # and displaying the raw numpy array.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels='RGB', width=1200)

        processed += 1
        frame_idx += 1
        st.session_state['live_frame_idx'] = frame_idx
        st.session_state['live_processed'] = processed

        # Update info
        elapsed = time.time() - start
        info.text(f"Frames: {processed}  Elapsed: {elapsed:.1f}s  (press Ctrl+C in terminal to stop)")

        # Sleep to approximate original frame rate minus processing time
        t1 = time.time()
        proc_time = t1 - t0
        to_sleep = max(0.0, delay - proc_time)
        time.sleep(to_sleep)

    # release capture and clear session state keys when done
    try:
        cap.release()
    except Exception:
        pass
    st.session_state.pop('cap_live', None)
    st.session_state.pop('live_video_path', None)
    st.session_state.pop('live_frame_idx', None)
    st.session_state.pop('live_processed', None)
    elapsed = time.time() - start
    return processed, elapsed


def process_one_frame_and_continue(model, input_path):
    """Process exactly one frame (respecting frame_skip and current session state) and display it.
    Then return a flag indicating whether to continue (auto-rerun) or stop.
    This function uses `st.session_state` to persist the VideoCapture and frame index across reruns.
    """
    # session keys: cap, frame_idx, processed
    if 'cap' not in st.session_state or st.session_state.get('video_path') != str(input_path):
        # initialize capture
        try:
            st.session_state['cap'] = cv2.VideoCapture(str(input_path))
            st.session_state['frame_idx'] = 0
            st.session_state['processed'] = 0
            st.session_state['video_path'] = str(input_path)
        except Exception as e:
            st.error(f"Failed to open video: {e}")
            return False

    cap = st.session_state['cap']
    if not cap or not cap.isOpened():
        st.error("Video capture is not available.")
        return False

    # read session-controlled params
    current_conf = float(st.session_state.get('confidence', 0.4))
    current_skip = int(st.session_state.get('frame_skip', 1))

    # attempt to read next valid frame
    max_attempts = 10
    attempts = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # loop file if possible
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return False

        if st.session_state['frame_idx'] % max(1, current_skip) != 0:
            st.session_state['frame_idx'] += 1
            attempts += 1
            if attempts > 1000:
                return False
            continue
        break

    t0 = time.time()

    # Run detection using ultralytics for bboxes
    results = model.predict(frame, conf=current_conf, verbose=False)

    rects = []
    if len(results) > 0:
        res = results[0]
        boxes = res.boxes
        if boxes is not None:
            for box in boxes:
                try:
                    cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confscore = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0

                    # determine drowning using optional DrownDetect model if available
                    is_drown = False
                    if HAS_DROWN and DROWN_MODEL is not None:
                        try:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            augimg = DROWN_AUG(image=np.array(pil_image))['image']
                            arr = np.transpose(augimg, (2,0,1)).astype(np.float32)
                            tensor = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
                            with torch.no_grad():
                                out = DROWN_MODEL(tensor)
                                _, pred = torch.max(out.data, 1)
                                label_name = DROWN_LB.classes_[pred]
                                if label_name == 'drowning':
                                    is_drown = True
                        except Exception:
                            is_drown = False
                    else:
                        if cls == 0:
                            is_drown = True

                    # Respect show toggles from sidebar
                    show_blue = st.session_state.get('show_blue', True)
                    show_red = st.session_state.get('show_red', True)
                    if is_drown and show_red:
                        color = (0, 0, 255)  # red
                        label = f"drowning:{confscore:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        rects.append((x1, y1, x2, y2))
                    elif not is_drown and show_blue:
                        color = (255, 0, 0)  # blue (BGR)
                        label = f"safe:{confscore:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        rects.append((x1, y1, x2, y2))
                except Exception:
                    continue

    # Convert and display
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder = st.empty()
    placeholder.image(rgb, channels='RGB', width=1200)

    st.session_state['frame_idx'] += 1
    st.session_state['processed'] += 1

    # continue auto-run if live_mode still enabled
    return True


def main():
    st.title("Drowning Detection Dashboard")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path (leave empty to use ./model.pt)", value="")
    # Live settings (update immediately)
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, float(st.session_state.get('confidence', 0.4)), 0.01, key='confidence')
    frame_skip = st.sidebar.number_input("Process every Nth frame (1 = every frame)", min_value=1, max_value=30, value=int(st.session_state.get('frame_skip', 5)), key='frame_skip')
    show_blue = st.sidebar.checkbox("Show safe (blue) boxes", value=bool(st.session_state.get('show_blue', True)), key='show_blue')
    show_red = st.sidebar.checkbox("Show drowning (red) boxes", value=bool(st.session_state.get('show_red', True)), key='show_red')

    model = load_model(model_path if model_path else None)
    if model is None:
        return

    # Start live streaming
    placeholder = st.empty()
    try:
        processed, elapsed = stream_process_video(model, TAPO_STREAM_URL, conf=st.session_state.get('confidence', 0.4), frame_skip=st.session_state.get('frame_skip', 5), placeholder=placeholder)
    except Exception as e:
        st.error(f"Error during live playback: {e}")


if __name__ == '__main__':
    main()
