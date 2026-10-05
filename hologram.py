import cv2
import mediapipe as mp
import math

# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

# -----------------------------
# MEDIAPIPE
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# -----------------------------
# HOLOGRAM OBJECT
# -----------------------------

object_x = 640
object_y = 360

object_size = 120

grabbed = False

frame_number = 0


# -----------------------------
# DRAW HOLOGRAM CUBE
# -----------------------------

def draw_cube(frame, x, y, size):

    depth = 35

    x1 = int(x - size / 2)
    y1 = int(y - size / 2)

    x2 = int(x + size / 2)
    y2 = int(y + size / 2)

    # Back face
    bx1 = x1 + depth
    by1 = y1 - depth

    bx2 = x2 + depth
    by2 = y2 - depth

    # Front square
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 200, 50),
        2
    )

    # Back square
    cv2.rectangle(
        frame,
        (bx1, by1),
        (bx2, by2),
        (255, 200, 50),
        2
    )

    # Connecting lines
    cv2.line(frame, (x1, y1), (bx1, by1), (255, 200, 50), 2)
    cv2.line(frame, (x2, y1), (bx2, by1), (255, 200, 50), 2)
    cv2.line(frame, (x1, y2), (bx1, by2), (255, 200, 50), 2)
    cv2.line(frame, (x2, y2), (bx2, by2), (255, 200, 50), 2)


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = frame_number * 33

    frame_number += 1

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    finger_x = None
    finger_y = None

    # -----------------------------
    # HAND DETECTION
    # -----------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Index finger
        index = hand[8]

        # Thumb
        thumb = hand[4]

        h, w, _ = frame.shape

        finger_x = int(index.x * w)
        finger_y = int(index.y * h)

        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        # Draw landmarks
        for landmark in hand:

            lx = int(landmark.x * w)
            ly = int(landmark.y * h)

            cv2.circle(
                frame,
                (lx, ly),
                4,
                (0, 255, 0),
                -1
            )

        # -----------------------------
        # PINCH DETECTION
        # -----------------------------

        distance = math.sqrt(
            (finger_x - thumb_x) ** 2 +
            (finger_y - thumb_y) ** 2
        )

        if distance < 50:

            if not grabbed:

                # Only grab when finger is near object
                object_distance = math.sqrt(
                    (finger_x - object_x) ** 2 +
                    (finger_y - object_y) ** 2
                )

                if object_distance < object_size:

                    grabbed = True

            if grabbed:

                object_x = finger_x
                object_y = finger_y

        else:

            grabbed = False

        # Draw finger
        cv2.circle(
            frame,
            (finger_x, finger_y),
            12,
            (255, 255, 255),
            -1
        )

        # Draw connection between thumb and finger
        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (finger_x, finger_y),
            (255, 255, 255),
            2
        )

    # -----------------------------
    # HOLOGRAM CUBE
    # -----------------------------

    draw_cube(
        frame,
        object_x,
        object_y,
        object_size
    )

    # -----------------------------
    # UI
    # -----------------------------

    if grabbed:

        status = "GRABBING"

    else:

        status = "PINCH TO GRAB"

    cv2.putText(
        frame,
        "HOLOGRAM CONTROL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 200, 50),
        2
    )

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 50),
        2
    )

    cv2.imshow(
        "Hologram Interface",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

camera.release()

landmarker.close()

cv2.destroyAllWindows()