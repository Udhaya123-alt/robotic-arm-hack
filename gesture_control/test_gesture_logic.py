from types import SimpleNamespace

from gesture_camera import axis_value, classify_hand


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def synthetic_hand(extended_count):
    points = [point(0.5, 0.8) for _ in range(21)]
    points[5], points[17] = point(0.42, 0.58), point(0.62, 0.58)
    for n, (tip, base) in enumerate(zip((4, 8, 12, 16, 20), (2, 6, 10, 14, 18))):
        x = 0.25 if n == 0 else 0.43 + n * 0.045
        points[base] = point(x, 0.62)
        points[tip] = point(x, 0.35 if n < extended_count else 0.70)
    return points


def main():
    assert classify_hand(synthetic_hand(0))[0] == "FIST: HOLD"
    assert classify_hand(synthetic_hand(1))[1] == "shoulder_pan"
    assert classify_hand(synthetic_hand(5))[1] == "wrist_roll"
    assert axis_value((0.5, 0.5), "horizontal") == 0.0
    assert axis_value((0.9, 0.5), "horizontal") == 1.0
    assert axis_value((0.5, 0.1), "vertical") == 1.0
    print("gesture logic tests passed")


if __name__ == "__main__":
    main()
