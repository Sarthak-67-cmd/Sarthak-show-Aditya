import cv2
import numpy as np
import mediapipe as mp
import os
import math
import time


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "hand_landmarker.task"

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720

PINCH_THRESHOLD = 0.055

CUBE_SIZE = 120

SMOOTHING = 0.35


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    print()
    print("ERROR: hand_landmarker.task not found.")
    print()
    print("Make sure it is here:")
    print()
    print(
        os.path.abspath(MODEL_FILE)
    )

    input("Press Enter to exit...")
    raise SystemExit


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print("ERROR: Could not open camera.")

    input("Press Enter to exit...")

    raise SystemExit


# ============================================================
# CUBE
# ============================================================

cube_x = WIDTH // 2
cube_y = HEIGHT // 2

target_cube_x = cube_x
target_cube_y = cube_y

cube_grabbed = False


# ============================================================
# HELPERS
# ============================================================

def distance(a, b):

    dx = a.x - b.x
    dy = a.y - b.y

    return math.sqrt(
        dx * dx +
        dy * dy
    )


def draw_cube(
    image,
    x,
    y,
    size,
    grabbed
):

    half = size // 2

    # --------------------------------------------------------
    # Front face
    # --------------------------------------------------------

    p1 = (
        int(x - half),
        int(y - half)
    )

    p2 = (
        int(x + half),
        int(y - half)
    )

    p3 = (
        int(x + half),
        int(y + half)
    )

    p4 = (
        int(x - half),
        int(y + half)
    )

    # --------------------------------------------------------
    # Back face
    # --------------------------------------------------------

    offset = 35

    b1 = (
        p1[0] - offset,
        p1[1] - offset
    )

    b2 = (
        p2[0] - offset,
        p2[1] - offset
    )

    b3 = (
        p3[0] - offset,
        p3[1] - offset
    )

    b4 = (
        p4[0] - offset,
        p4[1] - offset
    )

    # --------------------------------------------------------
    # Front
    # --------------------------------------------------------

    cv2.line(
        image,
        p1,
        p2,
        (255, 255, 255),
        3
    )

    cv2.line(
        image,
        p2,
        p3,
        (255, 255, 255),
        3
    )

    cv2.line(
        image,
        p3,
        p4,
        (255, 255, 255),
        3
    )

    cv2.line(
        image,
        p4,
        p1,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # Back
    # --------------------------------------------------------

    cv2.line(
        image,
        b1,
        b2,
        (120, 120, 120),
        2
    )

    cv2.line(
        image,
        b2,
        b3,
        (120, 120, 120),
        2
    )

    cv2.line(
        image,
        b3,
        b4,
        (120, 120, 120),
        2
    )

    cv2.line(
        image,
        b4,
        b1,
        (120, 120, 120),
        2
    )

    # --------------------------------------------------------
    # Connecting edges
    # --------------------------------------------------------

    cv2.line(
        image,
        p1,
        b1,
        (180, 180, 180),
        2
    )

    cv2.line(
        image,
        p2,
        b2,
        (180, 180, 180),
        2
    )

    cv2.line(
        image,
        p3,
        b3,
        (180, 180, 180),
        2
    )

    cv2.line(
        image,
        p4,
        b4,
        (180, 180, 180),
        2
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    if grabbed:

        cv2.circle(
            image,
            (
                int(x),
                int(y)
            ),
            10,
            (255, 255, 255),
            -1
        )


# ============================================================
# MAIN
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    print()
    print("========================================")
    print("        HAND HOLOGRAM STARTED")
    print("========================================")
    print()
    print("Controls:")
    print()
    print("Pinch thumb + index = Grab cube")
    print("Move your hand = Move cube")
    print("ESC = Exit")
    print()

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Camera frame failed."
            )

            time.sleep(0.05)

            continue

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )

        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )

        hand_detected = False

        pinch = False

        index_x = 0
        index_y = 0

        # ----------------------------------------------------
        # Process hand
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand_detected = True

            landmarks = (
                result.hand_landmarks[0]
            )

            # Thumb tip = 4
            # Index tip = 8

            thumb = landmarks[4]

            index = landmarks[8]

            pinch_distance = distance(
                thumb,
                index
            )

            pinch = (
                pinch_distance
                < PINCH_THRESHOLD
            )

            index_x = int(
                index.x * WIDTH
            )

            index_y = int(
                index.y * HEIGHT
            )

            # ------------------------------------------------
            # Draw hand points
            # ------------------------------------------------

            for landmark in landmarks:

                px = int(
                    landmark.x * WIDTH
                )

                py = int(
                    landmark.y * HEIGHT
                )

                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (255, 255, 255),
                    -1
                )

            # ------------------------------------------------
            # Draw index finger
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    index_x,
                    index_y
                ),
                12,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Draw pinch line
            # ------------------------------------------------

            thumb_x = int(
                thumb.x * WIDTH
            )

            thumb_y = int(
                thumb.y * HEIGHT
            )

            cv2.line(
                frame,
                (
                    thumb_x,
                    thumb_y
                ),
                (
                    index_x,
                    index_y
                ),
                (255, 255, 255),
                2
            )

        # ====================================================
        # GRAB LOGIC
        # ====================================================

        cube_distance = math.sqrt(
            (
                index_x -
                cube_x
            ) ** 2
            +
            (
                index_y -
                cube_y
            ) ** 2
        )

        if (
            hand_detected
            and pinch
            and not cube_grabbed
            and cube_distance < CUBE_SIZE
        ):

            cube_grabbed = True

        if not pinch:

            cube_grabbed = False

        # ====================================================
        # MOVE CUBE
        # ====================================================

        if (
            cube_grabbed
            and hand_detected
        ):

            target_cube_x = index_x
            target_cube_y = index_y

        # ----------------------------------------------------
        # Smooth movement
        # ----------------------------------------------------

        cube_x += (
            target_cube_x -
            cube_x
        ) * SMOOTHING

        cube_y += (
            target_cube_y -
            cube_y
        ) * SMOOTHING

        cube_x = int(cube_x)
        cube_y = int(cube_y)

        # ====================================================
        # DRAW CUBE
        # ====================================================

        draw_cube(
            frame,
            cube_x,
            cube_y,
            CUBE_SIZE,
            cube_grabbed
        )

        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            "HAND HOLOGRAM",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        if cube_grabbed:

            status = "GRABBING"

        elif pinch:

            status = "PINCH"

        else:

            status = "READY"

        cv2.putText(
            frame,
            f"Status: {status}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Pinch the cube and move your hand",
            (25, HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Hand Hologram",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Hand hologram stopped.")