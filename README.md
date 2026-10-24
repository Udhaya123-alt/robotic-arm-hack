# SO-101 OpenCV hand-gesture controller

This project controls an assembled SO-101 follower arm with hand gestures. OpenCV and MediaPipe run the webcam on Windows, while a small TCP bridge sends safety-limited commands to [LeRobot](https://github.com/huggingface/lerobot) inside Ubuntu/WSL.

The camera starts in preview mode. No motor command is sent unless the bridge is running, the calibration file matches the selected robot ID, and the operator presses **Space** to arm the controller.

## Gesture map

| Gesture | Joint | Hand movement |
| --- | --- | --- |
| Fist | HOLD | None |
| 1 finger | Shoulder pan | Left/right |
| 2 fingers | Shoulder lift | Up/down |
| 3 fingers | Elbow flex | Up/down |
| 4 fingers | Wrist flex | Up/down |
| 5 fingers | Wrist roll | Left/right |
| Pinch | Gripper | Left/right |

## 1. Install and preview on Windows

Install Python 3, then run:

```bat
gesture_control\setup-windows.cmd
start-gesture-preview.cmd
```

The setup installs OpenCV, MediaPipe, and NumPy and downloads the official MediaPipe Hand Landmarker model. Preview mode displays gesture recognition without connecting to the arm.

Keys:

- `Space`: arm or disarm when a robot bridge is connected
- `R`: record recognized gestures
- `P`: save the arm's current pose through the bridge
- `Q` or `Esc`: quit

## 2. Prepare LeRobot in WSL

Install the official [LeRobot](https://github.com/huggingface/lerobot) environment and calibrate the SO-101 follower. Attach the controller to WSL and find its port:

```bash
lerobot-find-port
ls /dev/ttyACM* /dev/ttyUSB*
```

The bridge intentionally refuses to start when it cannot find the calibration JSON associated with `--robot-id`.

## 3. Start the robot bridge

From the activated LeRobot environment in WSL:

```bash
python /path/to/robotic-arm-hack/gesture_control/robot_bridge.py \
  --serial-port /dev/ttyACM0 \
  --robot-id YOUR_CALIBRATION_ID
```

Then start the camera client from Windows:

```powershell
py -3 .\gesture_control\gesture_camera.py --connect
```

Keep the arm clear and confirm the overlay recognizes the intended gesture before pressing Space. The bridge starts disarmed, limits each joint increment, has a heartbeat watchdog, and disconnects with torque disabled.

## Recordings

- Gesture observations: `gesture_control/recordings/gesture_*.jsonl`
- Robot commands and joint samples: `gesture_control/recordings/robot_*.csv`
- Poses saved with `P`: `gesture_control/recordings/saved_poses.json`

Recordings, calibration files, downloaded models, and Python cache files are ignored by Git.

## Tests

```powershell
py -3 .\gesture_control\test_gesture_logic.py
py -3 -m py_compile .\gesture_control\gesture_camera.py .\gesture_control\robot_bridge.py
```

The SO-101 hardware design is maintained in [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).
