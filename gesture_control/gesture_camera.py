#!/usr/bin/env python3
"""Windows webcam hand-gesture controller for an SO-101 follower arm.

Preview is the default. Robot messages are sent only with --connect and after
the operator presses SPACE to arm the UI.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"
CONNECTIONS = tuple(mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)
FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_BASES = (2, 6, 10, 14, 18)
JOINT_FOR_COUNT = {
    1: ("shoulder_pan", "horizontal"),
    2: ("shoulder_lift", "vertical"),
    3: ("elbow_flex", "vertical"),
    4: ("wrist_flex", "vertical"),
    5: ("wrist_roll", "horizontal"),
}


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_hand(landmarks) -> tuple[str, str | None, str | None, int]:
    """Return label, joint, axis, and extended-finger count."""
    palm_width = max(distance(landmarks[5], landmarks[17]), 0.04)
    pinch_ratio = distance(landmarks[4], landmarks[8]) / palm_width
    index_extended = distance(landmarks[8], landmarks[0]) > 1.18 * distance(landmarks[6], landmarks[0])
    if pinch_ratio < 0.34 and index_extended:
        return "PINCH: gripper", "gripper", "horizontal", -1

    wrist = landmarks[0]
    count = sum(
        distance(landmarks[tip], wrist) > 1.18 * distance(landmarks[base], wrist)
        for tip, base in zip(FINGER_TIPS, FINGER_BASES, strict=True)
    )
    if count == 0:
        return "FIST: HOLD", None, None, 0
    joint, axis = JOINT_FOR_COUNT[count]
    return f"{count} finger(s): {joint}", joint, axis, count


def hand_center(landmarks) -> tuple[float, float]:
    ids = (0, 5, 9, 13, 17)
    return (
        sum(landmarks[i].x for i in ids) / len(ids),
        sum(landmarks[i].y for i in ids) / len(ids),
    )


def axis_value(center: tuple[float, float], axis: str | None) -> float:
    if axis == "horizontal":
        value = (center[0] - 0.5) / 0.32
    elif axis == "vertical":
        value = (0.5 - center[1]) / 0.32
    else:
        return 0.0
    value = max(-1.0, min(1.0, value))
    return 0.0 if abs(value) < 0.28 else value


class RobotLink:
    def __init__(self, host: str, port: int, enabled: bool):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.sock: socket.socket | None = None
        self.last_attempt = 0.0

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def send(self, message: dict) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        if self.sock is None:
            if now - self.last_attempt < 1.0:
                return False
            self.last_attempt = now
            try:
                sock = socket.create_connection((self.host, self.port), timeout=0.25)
                sock.settimeout(0.25)
                self.sock = sock
            except OSError:
                return False
        try:
            payload = json.dumps(message, separators=(",", ":")) + "\n"
            self.sock.sendall(payload.encode("utf-8"))
            return True
        except OSError:
            self.close()
            return False

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


def draw_hand(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    for connection in CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (80, 220, 80), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 210, 255), -1)


def put_line(frame, text: str, row: int, color=(240, 240, 240), scale=0.62) -> None:
    cv2.putText(frame, text, (14, 28 + row * 27), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (14, 28 + row * 27), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def open_camera(index: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, _ = capture.read()
            if ok:
                return capture
        capture.release()
    raise RuntimeError(f"Could not open camera {index}. Close Camera/Teams/Zoom and try again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--connect", action="store_true", help="enable connection to the WSL robot bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--stable-seconds", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"MediaPipe model is missing: {args.model}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    capture = open_camera(args.camera)
    link = RobotLink(args.host, args.port, args.connect)
    armed = False
    recording = False
    gesture_log: deque[dict] = deque(maxlen=10000)
    stable_label = "NO HAND"
    candidate_label = stable_label
    candidate_since = time.monotonic()
    last_send = 0.0
    last_heartbeat = 0.0
    started = time.monotonic()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames.")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.monotonic() - started) * 1000)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

                label, joint, axis, finger_count = "NO HAND", None, None, -2
                command_value = 0.0
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_hand(frame, landmarks)
                    label, joint, axis, finger_count = classify_hand(landmarks)
                    center = hand_center(landmarks)
                    command_value = axis_value(center, axis)
                    h, w = frame.shape[:2]
                    cv2.circle(frame, (int(center[0] * w), int(center[1] * h)), 12, (255, 80, 80), 2)

                now = time.monotonic()
                if label != candidate_label:
                    candidate_label = label
                    candidate_since = now
                elif now - candidate_since >= args.stable_seconds:
                    stable_label = candidate_label

                if stable_label != label:
                    joint = None
                    command_value = 0.0

                if armed and joint and command_value and now - last_send >= 0.10:
                    link.send({"type": "move", "joint": joint, "value": command_value, "gesture": stable_label})
                    last_send = now

                if now - last_heartbeat >= 0.20:
                    link.send({"type": "heartbeat", "armed": armed})
                    last_heartbeat = now

                if recording:
                    gesture_log.append({
                        "time": time.time(), "gesture": label, "stable": stable_label,
                        "joint": joint, "value": round(command_value, 4), "fingers": finger_count,
                    })

                status = "ARMED" if armed else "SAFE / PREVIEW"
                status_color = (40, 60, 255) if armed else (40, 220, 80)
                connection = "connected" if link.connected else ("waiting" if args.connect else "disabled")
                direction = "HOLD" if not command_value else ("+" if command_value > 0 else "-")
                put_line(frame, status, 0, status_color, 0.78)
                put_line(frame, f"Gesture: {label}", 1)
                put_line(frame, f"Stable: {stable_label} | command {direction} {abs(command_value):.2f}", 2)
                put_line(frame, f"Robot link: {connection} | recording: {'ON' if recording else 'OFF'}", 3)
                put_line(frame, "SPACE arm/disarm | R log | P save robot pose | Q quit", 4, (200, 230, 255), 0.55)
                cv2.imshow("SO-101 Hand Gesture Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    armed = not armed
                    link.send({"type": "arm", "armed": armed})
                elif key == ord("r"):
                    recording = not recording
                elif key == ord("p"):
                    link.send({"type": "record_pose", "label": f"pose_{int(time.time())}"})
    finally:
        link.send({"type": "arm", "armed": False})
        link.close()
        capture.release()
        cv2.destroyAllWindows()
        if gesture_log:
            log_dir = ROOT / "recordings"
            log_dir.mkdir(exist_ok=True)
            path = log_dir / time.strftime("gesture_%Y%m%d_%H%M%S.jsonl")
            path.write_text("\n".join(json.dumps(row) for row in gesture_log) + "\n", encoding="utf-8")
            print(f"Saved gesture log: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
