import cv2
import time
import os

print("========================================")
print("       HOLOGRAM ROOM SCANNER")
print("========================================")
print()
print("Controls:")
print("  S = Start scanning")
print("  Q = Finish scan")
print("  R = Reset")
print()
print("Starting camera...")
print()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    input("Press Enter to exit...")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames = []
scanning = False
last_capture = 0

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    h, w = frame.shape[:2]

    # Center marker
    cv2.circle(
        display,
        (w // 2, h // 2),
        8,
        (0, 255, 255),
        2
    )

    # Status
    if scanning:

        cv2.putText(
            display,
            "SCANNING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Captured frames: {len(frames)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Move camera SLOWLY",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Capture a frame every 0.7 seconds
        now = time.time()

        if now - last_capture > 0.7:

            # Resize for easier stitching
            small = cv2.resize(
                frame,
                (960, 540)
            )

            frames.append(small)

            last_capture = now

            print(
                f"Captured frame {len(frames)}"
            )

    else:

        cv2.putText(
            display,
            "READY - Press S",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Slowly rotate around the room",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Hologram Room Scanner",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # Start
    if key == ord("s"):

        scanning = True
        frames = []
        last_capture = 0

        print()
        print("========================================")
        print("SCAN STARTED")
        print("Move the camera slowly.")
        print("Keep some overlap between views.")
        print("Press Q when finished.")
        print("========================================")
        print()

    # Reset
    elif key == ord("r"):

        scanning = False
        frames = []

        print("Scan reset.")

    # Finish
    elif key == ord("q"):

        break


camera.release()
cv2.destroyAllWindows()


# ==========================================
# STITCH PANORAMA
# ==========================================

print()
print("========================================")
print("PROCESSING ROOM SCAN")
print("========================================")
print()

if len(frames) < 3:

    print(
        f"Not enough frames: {len(frames)}"
    )

    print(
        "Scan for at least 5-10 seconds."
    )

    input("Press Enter to exit...")
    exit()


print(
    f"Trying to stitch {len(frames)} frames..."
)

stitcher = cv2.Stitcher_create()

stitcher.setPanoConfidenceThresh(0.5)

status, panorama = stitcher.stitch(frames)


if status == cv2.Stitcher_OK:

    # Crop black borders

    gray = cv2.cvtColor(
        panorama,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        panorama = panorama[
            y:y+h,
            x:x+w
        ]

    cv2.imwrite(
        "room_scan.jpg",
        panorama
    )

    print()
    print("========================================")
    print("          SCAN COMPLETE")
    print("========================================")
    print()
    print("Saved:")
    print("room_scan.jpg")
    print()

    cv2.imshow(
        "ROOM SCAN RESULT",
        panorama
    )

    print("Press any key to close.")

    cv2.waitKey(0)

else:

    print()
    print("========================================")
    print("       STITCHING FAILED")
    print("========================================")
    print()

    print("OpenCV error code:", status)

    print()
    print("Try again with:")
    print("1. Better lighting")
    print("2. Slower camera movement")
    print("3. Lots of visible objects/details")
    print("4. More overlap between views")
    print("5. Avoid pointing at plain walls")
    print()

input("Press Enter to exit...")