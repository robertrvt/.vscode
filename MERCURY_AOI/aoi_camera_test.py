"""MERCURY AOI - Step 1: Logitech C920 snapshot capture.

Windows / VS Code setup:
    python -m pip install opencv-python
    python aoi_camera_test.py

If the laptop camera opens, try:
    python aoi_camera_test.py --camera 1

Click the preview window before using these keys:
    R      Save a reference photo of a known good part.
    S      Save an inspection photo.
    C      Open Windows camera controls (DirectShow only).
    Q/ESC  Quit.

Photos go into an "aoi_captures" folder beside this script. Each gets a
unique filename. Saved photos contain the original camera pixels; the
smaller preview and its text are never saved into the photo.

This first step captures photos. Comparison and pass/fail rules come later.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

try:
    import cv2
except ImportError:
    raise SystemExit(
        "OpenCV is missing. Run: python -m pip install opencv-python"
    ) from None


REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080
WINDOW = "MERCURY AOI - Camera Test"


def save_snapshot(frame, output_dir, kind):
    """Write an unaltered frame as PNG, without replacing earlier captures."""
    folder = output_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"{kind}_{stamp}_{width}x{height}.png"
    # Encode before writing to support Windows paths containing Unicode.
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise OSError("OpenCV could not encode the snapshot.")
    with path.open("xb") as photo:
        photo.write(encoded.tobytes())
    return path


def make_preview(frame, message):
    """Resize a separate copy for display, keeping the snapshot untouched."""
    height, width = frame.shape[:2]
    scale = min(960 / width, 540 / height, 1.0)
    preview = cv2.resize(
        frame, (round(width * scale), round(height * scale))
    )
    lines = [
        f"Actual capture: {width} x {height}",
        "R: reference | S: inspection | C: controls | Q: quit",
        message,
    ]
    for index, line in enumerate(lines):
        position = (10, 24 + 26 * index)
        # A dark outline keeps the text readable against bright objects.
        cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index; try 1 or 2 if 0 opens another camera.")
    parser.add_argument("--backend", choices=("dshow", "msmf", "any"),
                        default="dshow" if sys.platform == "win32" else "any",
                        help="On Windows, try msmf if dshow cannot capture.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "aoi_captures",
                        help="Optional directory for captured photos.")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be 0 or a positive integer.")
    backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                "any": cv2.CAP_ANY}
    capture = cv2.VideoCapture(args.camera, backends[args.backend])
    output_dir = args.output.expanduser().resolve()

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera {args.camera}. Close Camera/Teams/Zoom, "
                "check Windows camera permissions, then try --camera 1 "
                "or --camera 2. On Windows you can also try --backend msmf."
            )

        # These are requests. Drivers may deliver a different resolution.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, 30)
        print(f"OpenCV {cv2.__version__}; camera {args.camera}; "
              f"backend {capture.getBackendName()}")
        print(f"Photos will be saved to: {output_dir}")
        print("Click the preview. R = reference; S = inspection; "
              "C = camera controls; Q/Esc = quit.")
        print("Check that this is the C920 and wait for focus/brightness "
              "to settle before saving.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        message = "Wait for focus/brightness to settle."
        message_until = time.monotonic() + 3
        failed_reads = 0
        last_resolution = None

        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                failed_reads += 1
                if failed_reads >= 20:
                    raise RuntimeError(
                        "The camera opened but stopped providing images. "
                        "Check the USB connection and close other camera apps. "
                        "On Windows, retry with --backend msmf."
                    )
                if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            failed_reads = 0
            height, width = frame.shape[:2]
            if (width, height) != last_resolution:
                print(f"Actual image size: {width} x {height}")
                if (width, height) != (REQUESTED_WIDTH, REQUESTED_HEIGHT):
                    print("NOTE: The camera is delivering less/different "
                          "resolution than the requested 1920 x 1080. "
                          "Photos will use the actual size shown above.")
                last_resolution = (width, height)

            if time.monotonic() > message_until:
                message = "Position the part and keep the camera steady."
            cv2.imshow(WINDOW, make_preview(frame, message))
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R"), ord("s"), ord("S")):
                kind = "reference" if key in (ord("r"), ord("R")) else "inspection"
                try:
                    path = save_snapshot(frame, output_dir, kind)
                    print(f"Saved: {path}")
                    message = f"Saved {kind} photo ({width} x {height})."
                except (OSError, cv2.error) as error:
                    print(f"Snapshot was not saved: {error}")
                    message = "Save failed. See the terminal."
                message_until = time.monotonic() + 3
            elif key in (ord("c"), ord("C")):
                if capture.getBackendName().upper() == "DSHOW":
                    print("Opening camera controls. If available, adjust focus "
                          "and exposure, then disable their Auto boxes. "
                          "Use the same settings for reference and inspection.")
                    capture.set(cv2.CAP_PROP_SETTINGS, 0)
                    message = "Wait for settings to settle before saving."
                else:
                    print("The camera controls dialog needs Windows DirectShow. "
                          "Try --backend dshow on Windows.")
                    message = "Camera controls unavailable with this backend."
                message_until = time.monotonic() + 3
    except (RuntimeError, cv2.error) as error:
        print(f"Camera test stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Camera test stopped.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
