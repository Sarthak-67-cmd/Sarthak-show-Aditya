import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()

# MediaPipe Tasks API
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

print("Camera started!")
print("Move your hand in front of the camera.")
print("Press Q to quit.")

frame_number = 0

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

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            # Index fingertip = landmark 8
            index = hand[8]

            h, w, _ = frame.shape

            x = int(index.x * w)
            y = int(index.y * h)

            # Draw fingertip
            cv2.circle(
                frame,
                (x, y),
                15,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"Finger: {x}, {y}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Draw hand landmarks
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

    cv2.imshow("Hologram Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()