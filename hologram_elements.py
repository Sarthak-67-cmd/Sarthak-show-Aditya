import cv2
import numpy as np
import mediapipe as mp
import math

# ============================================================
# HOLOGRAM ELEMENTS
# Cube -> Zoom -> Reveal Elements -> Grab Element -> Move
# ============================================================

MODEL_PATH = "hand_landmarker.task"

# ------------------------------------------------------------
# MediaPipe Hand Landmarker
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------------------------------------------------------
# State
# ------------------------------------------------------------

cube_x = 640
cube_y = 360

cube_size = 170

zoom = 1.0
target_zoom = 1.0

cube_grabbed = False
element_grabbed = -1

elements_visible = False

previous_pinch = False

# Element positions relative to cube
elements = [
    {
        "name": "ELEMENT A",
        "offset": (-55, -55),
        "radius": 22,
        "color": (0, 180, 255)
    },
    {
        "name": "ELEMENT B",
        "offset": (55, -55),
        "radius": 22,
        "color": (255, 100, 100)
    },
    {
        "name": "ELEMENT C",
        "offset": (-55, 55),
        "radius": 22,
        "color": (100, 255, 120)
    },
    {
        "name": "ELEMENT D",
        "offset": (55, 55),
        "radius": 22,
        "color": (180, 100, 255)
    }
]

# Once an element is grabbed, it gets its own screen position.
element_positions = [None, None, None, None]

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def draw_glow_circle(frame, x, y, radius, color):
    # Simple layered glow effect
    for r in range(radius + 18, radius, -4):
        alpha = (radius + 18 - r) / 18.0

        overlay = frame.copy()

        cv2.circle(
            overlay,
            (int(x), int(y)),
            r,
            color,
            2
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.12 * alpha,
            frame,
            1 - 0.12 * alpha,
            0
        )

    cv2.circle(
        frame,
        (int(x), int(y)),
        radius,
        color,
        2
    )

    cv2.circle(
        frame,
        (int(x), int(y)),
        max(3, radius // 5),
        color,
        -1
    )


def draw_cube(frame, x, y, size, zoom_level):
    s = int(size * zoom_level / 2)

    # Perspective cube points
    front = np.array([
        [x - s, y - s],
        [x + s, y - s],
        [x + s, y + s],
        [x - s, y + s]
    ])

    depth = int(s * 0.45)

    back = np.array([
        [x - s + depth, y - s - depth],
        [x + s + depth, y - s - depth],
        [x + s + depth, y + s - depth],
        [x - s + depth, y + s - depth]
    ])

    # Front face
    for i in range(4):
        p1 = tuple(front[i])
        p2 = tuple(front[(i + 1) % 4])

        cv2.line(
            frame,
            p1,
            p2,
            (255, 220, 80),
            2
        )

    # Back face
    for i in range(4):
        p1 = tuple(back[i])
        p2 = tuple(back[(i + 1) % 4])

        cv2.line(
            frame,
            p1,
            p2,
            (120, 160, 255),
            1
        )

    # Connecting edges
    for i in range(4):
        cv2.line(
            frame,
            tuple(front[i]),
            tuple(back[i]),
            (180, 200, 255),
            2
        )

    # Center
    cv2.circle(
        frame,
        (int(x), int(y)),
        5,
        (255, 255, 255),
        -1
    )


def cube_hit(px, py):
    half = cube_size * zoom / 2

    return (
        cube_x - half < px < cube_x + half and
        cube_y - half < py < cube_y + half
    )


def get_element_position(index):
    if element_positions[index] is not None:
        return element_positions[index]

    ox, oy = elements[index]["offset"]

    return (
        cube_x + ox * zoom,
        cube_y + oy * zoom
    )


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

print()
print("========================================")
print("       HOLOGRAM ELEMENT SYSTEM")
print("========================================")
print()
print("1. Pinch the cube")
print("2. Move your hand toward the camera")
print("3. Cube zooms")
print("4. Elements appear")
print("5. Pinch an element")
print("6. Move your hand")
print("7. Release to place it")
print()
print("ESC = Exit")
print()

frame_number = 0

while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame failed.")
        continue

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = frame_number * 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    frame_number += 1

    index_x = None
    index_y = None

    thumb_x = None
    thumb_y = None

    pinch = False

    # --------------------------------------------------------
    # Hand landmarks
    # --------------------------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index = hand[8]
        thumb = hand[4]

        index_x = int(index.x * width)
        index_y = int(index.y * height)

        thumb_x = int(thumb.x * width)
        thumb_y = int(thumb.y * height)

        pinch_distance = distance(
            index_x,
            index_y,
            thumb_x,
            thumb_y
        )

        pinch = pinch_distance < 45

        # Draw hand points
        cv2.circle(
            frame,
            (index_x, index_y),
            8,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            8,
            (255, 180, 0),
            -1
        )

        cv2.line(
            frame,
            (index_x, index_y),
            (thumb_x, thumb_y),
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Detect pinch START
    # --------------------------------------------------------

    pinch_started = pinch and not previous_pinch
    pinch_released = not pinch and previous_pinch

    # --------------------------------------------------------
    # Select cube
    # --------------------------------------------------------

    if pinch_started and index_x is not None:

        if cube_hit(index_x, index_y):

            cube_grabbed = True
            element_grabbed = -1

    # --------------------------------------------------------
    # Cube grabbed
    # --------------------------------------------------------

    if cube_grabbed and pinch and index_x is not None:

        cube_x = index_x
        cube_y = index_y

        # Estimate zoom from hand distance to camera
        # based on the apparent hand size.
        #
        # This is intentionally bounded so the cube
        # doesn't become enormous.

        hand_width = 0

        if result.hand_landmarks:

            wrist = result.hand_landmarks[0][0]
            middle = result.hand_landmarks[0][12]

            wx = wrist.x * width
            wy = wrist.y * height

            mx = middle.x * width
            my = middle.y * height

            hand_width = distance(
                wx,
                wy,
                mx,
                my
            )

        if hand_width > 0:

            target_zoom = np.clip(
                hand_width / 70.0,
                1.0,
                3.0
            )

    # --------------------------------------------------------
    # Release cube
    # --------------------------------------------------------

    if pinch_released and cube_grabbed:

        cube_grabbed = False

        if zoom > 1.45:
            elements_visible = True

    # --------------------------------------------------------
    # Smooth zoom
    # --------------------------------------------------------

    zoom += (target_zoom - zoom) * 0.12

    # --------------------------------------------------------
    # Select an element
    # --------------------------------------------------------

    if elements_visible and pinch_started and index_x is not None:

        for i in range(len(elements)):

            ex, ey = get_element_position(i)

            if distance(
                index_x,
                index_y,
                ex,
                ey
            ) < elements[i]["radius"] * 2:

                element_grabbed = i

                # Save its current position
                element_positions[i] = (
                    index_x,
                    index_y
                )

                break

    # --------------------------------------------------------
    # Move selected element
    # --------------------------------------------------------

    if (
        elements_visible
        and element_grabbed >= 0
        and pinch
        and index_x is not None
    ):

        element_positions[element_grabbed] = (
            index_x,
            index_y
        )

    # --------------------------------------------------------
    # Release element
    # --------------------------------------------------------

    if pinch_released and element_grabbed >= 0:

        element_grabbed = -1

    # --------------------------------------------------------
    # Draw cube
    # --------------------------------------------------------

    draw_cube(
        frame,
        cube_x,
        cube_y,
        cube_size,
        zoom
    )

    # --------------------------------------------------------
    # Draw elements
    # --------------------------------------------------------

    if elements_visible:

        for i, element in enumerate(elements):

            ex, ey = get_element_position(i)

            draw_glow_circle(
                frame,
                ex,
                ey,
                element["radius"],
                element["color"]
            )

            # Element label
            cv2.putText(
                frame,
                element["name"],
                (
                    int(ex - 45),
                    int(ey + element["radius"] + 25)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                element["color"],
                1,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if element_grabbed >= 0:

        status = (
            "HOLDING: "
            + elements[element_grabbed]["name"]
        )

    elif cube_grabbed:

        status = "GRABBING CUBE"

    elif elements_visible:

        status = "PINCH AN ELEMENT"

    else:

        status = "PINCH THE CUBE"

    cv2.rectangle(
        frame,
        (15, 15),
        (360, 55),
        (15, 15, 15),
        -1
    )

    cv2.putText(
        frame,
        status,
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Pinch = Grab / Release",
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "ESC = Exit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(
        "Hologram Elements",
        frame
    )

    previous_pinch = pinch

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

camera.release()
cv2.destroyAllWindows()
landmarker.close()

print()
print("Hologram element system stopped.")