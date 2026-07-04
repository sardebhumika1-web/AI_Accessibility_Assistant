import time
import cv2
import av

from streamlit_webrtc import VideoProcessorBase

from config import CONFIDENCE_THRESHOLD
from utils import detect_objects, speak, add_history


class VideoProcessor(VideoProcessorBase):
    """Real-time AI Accessibility Video Processor"""

    def __init__(self):
        self.last_spoken = ""
        self.last_history = ""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            annotated, detections = detect_objects(
                image,
                confidence=CONFIDENCE_THRESHOLD
            )
        except Exception as e:
            cv2.putText(
                image,
                f"Detection Error: {e}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        if detections:
            labels = sorted(set(d["label"] for d in detections))
            current = ", ".join(labels)

            if current != self.last_spoken:
                message = [
                    f"{obj['label']} on your {obj['direction']}"
                    for obj in detections[:3]
                ]
                speak(". ".join(message))
                self.last_spoken = current

            if current != self.last_history:
                add_history(detections)
                self.last_history = current

        cv2.putText(
            annotated,
            f"Objects: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )