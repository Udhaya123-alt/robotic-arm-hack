#!/usr/bin/env python3
"""Safety-gated TCP bridge from Windows hand gestures to an SO-101 in WSL."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import time
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


BODY_JOINTS = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
ALL_JOINTS = BODY_JOINTS | {"gripper"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", required=True, help="for example /dev/ttyACM0")
    parser.add_argument("--robot-id", required=True, help="must match the ID used during calibration")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--body-step", type=float, default=0.7, help="maximum degrees per accepted message")
    parser.add_argument("--gripper-step", type=float, default=1.0, help="maximum normalized units per message")
    parser.add_argument("--watchdog", type=float, default=0.65)
    return parser.parse_args()


def save_pose(path: Path, label: str, observation: dict[str, float]) -> None:
    rows = []
    if path.is_file():
        rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"label": label, "time": time.time(), "joints": observation})
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pose '{label}' to {path}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(__file__).resolve().parent / "recordings"
    output_dir.mkdir(exist_ok=True)

    config = SO101FollowerConfig(
        port=args.serial_port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target={**{joint: 1.0 for joint in BODY_JOINTS}, "gripper": 1.5},
        disable_torque_on_disconnect=True,
    )
    robot = SO101Follower(config)
    if not robot.calibration_fpath.is_file():
        raise FileNotFoundError(
            f"Calibration not found: {robot.calibration_fpath}\n"
            "Use the exact --robot-id that you used during calibration."
        )

    print(f"Using calibration: {robot.calibration_fpath}")
    print("Connecting with calibration prompts disabled...")
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        raise RuntimeError("The arm and calibration file do not match. Movement was not enabled.")

    log_path = output_dir / time.strftime("robot_%Y%m%d_%H%M%S.csv")
    pose_path = output_dir / "saved_poses.json"
    fieldnames = ["time", "gesture", "joint", "command"] + [f"{j}.pos" for j in sorted(ALL_JOINTS)]
    last_heartbeat = 0.0
    armed = False

    try:
        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((args.host, args.port))
                server.listen(1)
                print(f"Bridge ready on TCP {args.host}:{args.port}; waiting for Windows camera client.")
                while True:
                    connection, address = server.accept()
                    print(f"Camera client connected from {address[0]}. Movement starts DISARMED.")
                    armed = False
                    last_heartbeat = time.monotonic()
                    connection.settimeout(0.20)
                    buffer = b""
                    try:
                        while True:
                            if time.monotonic() - last_heartbeat > args.watchdog:
                                armed = False
                            try:
                                block = connection.recv(4096)
                                if not block:
                                    break
                                buffer += block
                            except TimeoutError:
                                continue
                            while b"\n" in buffer:
                                raw, buffer = buffer.split(b"\n", 1)
                                message = json.loads(raw.decode("utf-8"))
                                kind = message.get("type")
                                if kind == "heartbeat":
                                    last_heartbeat = time.monotonic()
                                    if not message.get("armed", False):
                                        armed = False
                                elif kind == "arm":
                                    armed = bool(message.get("armed"))
                                    last_heartbeat = time.monotonic()
                                    print("ARMED" if armed else "DISARMED")
                                elif kind == "record_pose":
                                    obs = robot.get_observation()
                                    pose = {key: float(value) for key, value in obs.items() if key.endswith(".pos")}
                                    save_pose(pose_path, str(message.get("label", "pose")), pose)
                                elif kind == "move" and armed:
                                    joint = str(message.get("joint", ""))
                                    value = max(-1.0, min(1.0, float(message.get("value", 0.0))))
                                    if joint not in ALL_JOINTS or value == 0.0:
                                        continue
                                    obs = robot.get_observation()
                                    action = {key: float(val) for key, val in obs.items() if key.endswith(".pos")}
                                    step = args.gripper_step if joint == "gripper" else args.body_step
                                    key = f"{joint}.pos"
                                    action[key] += step * value
                                    if joint == "gripper":
                                        action[key] = max(0.0, min(100.0, action[key]))
                                    sent = robot.send_action(action)
                                    writer.writerow({
                                        "time": time.time(), "gesture": message.get("gesture", ""),
                                        "joint": joint, "command": value, **sent,
                                    })
                                    log_file.flush()
                    except (ConnectionError, json.JSONDecodeError) as error:
                        print(f"Client disconnected: {error}")
                    finally:
                        armed = False
                        connection.close()
                        print("DISARMED; waiting for reconnect.")
    except KeyboardInterrupt:
        print("Stopping bridge.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print(f"Robot log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
