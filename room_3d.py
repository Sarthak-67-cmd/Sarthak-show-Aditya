import cv2
import numpy as np
import time
import os


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

WIDTH = 1280
HEIGHT = 720

MAX_FEATURES = 2500
MAX_POINTS = 15000
MAX_FAILED_FRAMES = 30

MIN_MATCHES = 40
MIN_INLIERS = 25

OUTPUT_FILE = "room_pointcloud.ply"


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    input("Press Enter to exit...")
    raise SystemExit


success, frame = camera.read()

if not success:
    print("ERROR: Camera opened but first frame failed.")
    camera.release()
    input("Press Enter to exit...")
    raise SystemExit


actual_height, actual_width = frame.shape[:2]

print()
print("========================================")
print("       ROOM 3D MAPPING SYSTEM")
print("========================================")
print()
print(f"Camera resolution: {actual_width} x {actual_height}")
print()
print("Controls:")
print("  S = Start mapping")
print("  R = Reset mapping")
print("  Q = Finish and save")
print()
print("Move the camera slowly around the room.")
print("Keep walls, furniture and other details visible.")
print()


# ============================================================
# CAMERA INTRINSICS
# ============================================================

focal_length = actual_width * 0.9

cx = actual_width / 2
cy = actual_height / 2

K = np.array(
    [
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1],
    ],
    dtype=np.float64
)


# ============================================================
# FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=MAX_FEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    fastThreshold=12
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# ============================================================
# MAPPING VARIABLES
# ============================================================

mapping_started = False

previous_gray = None
previous_keypoints = None
previous_descriptors = None

camera_position = np.zeros(3, dtype=np.float64)

camera_rotation = np.eye(3, dtype=np.float64)

trajectory = []

points_3d = []

failed_frames = 0

frame_count = 0


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_mapping():

    global mapping_started
    global previous_gray
    global previous_keypoints
    global previous_descriptors
    global camera_position
    global camera_rotation
    global trajectory
    global points_3d
    global failed_frames
    global frame_count

    mapping_started = False

    previous_gray = None
    previous_keypoints = None
    previous_descriptors = None

    camera_position = np.zeros(3, dtype=np.float64)

    camera_rotation = np.eye(3, dtype=np.float64)

    trajectory = []

    points_3d = []

    failed_frames = 0
    frame_count = 0

    print()
    print("Mapping reset.")
    print()


# ============================================================
# SAVE PLY
# ============================================================

def save_point_cloud(filename, points):

    if len(points) == 0:

        print()
        print("No 3D points available.")
        return False

    clean_points = []

    for point in points:

        if len(point) != 3:
            continue

        x, y, z = point

        if not np.isfinite(x):
            continue

        if not np.isfinite(y):
            continue

        if not np.isfinite(z):
            continue

        clean_points.append(
            (float(x), float(y), float(z))
        )

    if len(clean_points) == 0:

        print("No valid 3D points to save.")
        return False

    try:

        with open(filename, "w", encoding="utf-8") as file:

            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(
                f"element vertex {len(clean_points)}\n"
            )
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")

            for x, y, z in clean_points:

                file.write(
                    f"{x:.6f} {y:.6f} {z:.6f}\n"
                )

        print()
        print("========================================")
        print("       POINT CLOUD SAVED")
        print("========================================")
        print()
        print(f"File: {os.path.abspath(filename)}")
        print(f"Points: {len(clean_points)}")
        print()

        return True

    except Exception as error:

        print()
        print("ERROR saving point cloud:")
        print(error)

        return False


# ============================================================
# DRAW MAP
# ============================================================

def draw_map(points, path, current_position):

    map_width = 800
    map_height = 700

    canvas = np.zeros(
        (map_height, map_width, 3),
        dtype=np.uint8
    )

    center_x = map_width // 2
    center_y = map_height // 2

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid_size = 50

    for x in range(0, map_width, grid_size):

        cv2.line(
            canvas,
            (x, 0),
            (x, map_height),
            (40, 40, 40),
            1
        )

    for y in range(0, map_height, grid_size):

        cv2.line(
            canvas,
            (0, y),
            (map_width, y),
            (40, 40, 40),
            1
        )

    # --------------------------------------------------------
    # Coordinate scale
    # --------------------------------------------------------

    scale = 80

    # --------------------------------------------------------
    # 3D points
    # Top-down projection:
    #
    # X -> horizontal
    # Z -> vertical
    # --------------------------------------------------------

    if len(points) > 0:

        step = max(1, len(points) // 6000)

        for point in points[::step]:

            x = point[0]
            z = point[2]

            px = int(center_x + x * scale)
            py = int(center_y + z * scale)

            if 0 <= px < map_width and 0 <= py < map_height:

                canvas[py, px] = (0, 180, 0)

    # --------------------------------------------------------
    # Camera trajectory
    # --------------------------------------------------------

    if len(path) > 1:

        for i in range(1, len(path)):

            p1 = path[i - 1]
            p2 = path[i]

            x1 = int(center_x + p1[0] * scale)
            y1 = int(center_y + p1[2] * scale)

            x2 = int(center_x + p2[0] * scale)
            y2 = int(center_y + p2[2] * scale)

            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

    # --------------------------------------------------------
    # Current camera
    # --------------------------------------------------------

    camera_x = int(
        center_x + current_position[0] * scale
    )

    camera_y = int(
        center_y + current_position[2] * scale
    )

    cv2.circle(
        canvas,
        (camera_x, camera_y),
        8,
        (0, 0, 255),
        -1
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "3D ROOM MAP",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"3D Points: {len(points)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        f"Camera positions: {len(path)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        "RED = Camera",
        (20, map_height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        canvas,
        "GREEN = 3D Points",
        (20, map_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 180, 0),
        2
    )

    return canvas


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        failed_frames += 1

        print(
            f"WARNING: Camera frame failed "
            f"({failed_frames}/{MAX_FAILED_FRAMES})"
        )

        time.sleep(0.05)

        if failed_frames >= MAX_FAILED_FRAMES:

            print()
            print("Camera failed for too long.")
            print("Stopping safely.")
            break

        continue

    failed_frames = 0

    frame_count += 1

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # BEFORE MAPPING
    # ========================================================

    if not mapping_started:

        display = frame.copy()

        cv2.putText(
            display,
            "ROOM 3D MAPPING",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press S to start scanning",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Move camera slowly after starting",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Camera",
            display
        )

    # ========================================================
    # MAPPING
    # ========================================================

    else:

        keypoints, descriptors = orb.detectAndCompute(
            gray,
            None
        )

        if (
            previous_descriptors is not None
            and descriptors is not None
            and len(previous_descriptors) > 10
            and len(descriptors) > 10
        ):

            matches = matcher.match(
                previous_descriptors,
                descriptors
            )

            matches = sorted(
                matches,
                key=lambda x: x.distance
            )

            # Keep the strongest matches
            good_matches = matches[:150]

            if len(good_matches) >= MIN_MATCHES:

                pts_previous = np.float32(
                    [
                        previous_keypoints[m.queryIdx].pt
                        for m in good_matches
                    ]
                )

                pts_current = np.float32(
                    [
                        keypoints[m.trainIdx].pt
                        for m in good_matches
                    ]
                )

                # =================================================
                # Essential matrix
                # =================================================

                E, mask = cv2.findEssentialMat(
                    pts_previous,
                    pts_current,
                    K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:

                    try:

                        inlier_count, R, t, pose_mask = cv2.recoverPose(
                            E,
                            pts_previous,
                            pts_current,
                            K
                        )

                        if inlier_count >= MIN_INLIERS:

                            # =====================================
                            # Camera motion
                            # =====================================

                            translation = t.reshape(3)

                            # Scale is unknown with a normal webcam.
                            # Normalize to prevent huge jumps.
                            translation_norm = np.linalg.norm(
                                translation
                            )

                            if translation_norm > 0:

                                translation = (
                                    translation /
                                    translation_norm
                                )

                            # Small step size
                            translation *= 0.05

                            # Update world camera position
                            camera_position += (
                                camera_rotation @ translation
                            )

                            camera_rotation = (
                                camera_rotation @ R.T
                            )

                            trajectory.append(
                                camera_position.copy()
                            )

                            # =====================================
                            # Triangulation
                            # =====================================

                            P1 = K @ np.hstack(
                                (
                                    np.eye(3),
                                    np.zeros((3, 1))
                                )
                            )

                            P2 = K @ np.hstack(
                                (
                                    R,
                                    t
                                )
                            )

                            triangulated = cv2.triangulatePoints(
                                P1,
                                P2,
                                pts_previous.T,
                                pts_current.T
                            )

                            triangulated /= (
                                triangulated[3:4, :] + 1e-8
                            )

                            local_points = (
                                triangulated[:3, :]
                                .T
                            )

                            # =====================================
                            # Convert points into approximate
                            # world coordinates
                            # =====================================

                            world_points = (
                                camera_rotation
                                @ local_points.T
                            ).T

                            world_points += camera_position

                            # =====================================
                            # Filter bad points
                            # =====================================

                            for point in world_points:

                                x, y, z = point

                                if not np.isfinite(x):
                                    continue

                                if not np.isfinite(y):
                                    continue

                                if not np.isfinite(z):
                                    continue

                                # Remove extremely large points
                                distance = np.linalg.norm(point)

                                if distance > 20:
                                    continue

                                if distance < 0.01:
                                    continue

                                points_3d.append(
                                    point.astype(np.float64)
                                )

                            # =====================================
                            # Limit memory
                            # =====================================

                            if len(points_3d) > MAX_POINTS:

                                points_3d = points_3d[
                                    -MAX_POINTS:
                                ]

                    except cv2.error:
                        pass

        # ========================================================
        # Store current frame
        # ========================================================

        previous_gray = gray.copy()

        previous_keypoints = keypoints

        previous_descriptors = descriptors

        # ========================================================
        # Show camera
        # ========================================================

        display = frame.copy()

        cv2.putText(
            display,
            "MAPPING...",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"3D points: {len(points_3d)}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Camera positions: {len(trajectory)}",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "R = Reset    Q = Save & Quit",
            (25, actual_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Camera",
            display
        )

        # ========================================================
        # Show map
        # ========================================================

        map_image = draw_map(
            points_3d,
            trajectory,
            camera_position
        )

        cv2.imshow(
            "3D Room Map",
            map_image
        )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Start
    if key == ord("s"):

        if not mapping_started:

            mapping_started = True

            previous_gray = None
            previous_keypoints = None
            previous_descriptors = None

            camera_position = np.zeros(
                3,
                dtype=np.float64
            )

            camera_rotation = np.eye(
                3,
                dtype=np.float64
            )

            trajectory = []

            points_3d = []

            print()
            print("========================================")
            print("       3D MAPPING STARTED")
            print("========================================")
            print()
            print("Move the camera slowly.")
            print("Keep objects and wall details visible.")
            print()

    # Reset
    elif key == ord("r"):

        reset_mapping()

    # Quit
    elif key == ord("q"):

        break


# ============================================================
# FINISH
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("       3D MAPPING FINISHED")
print("========================================")
print()

print(
    f"Camera positions: {len(trajectory)}"
)

print(
    f"3D points created: {len(points_3d)}"
)


# ============================================================
# SAVE POINT CLOUD
# ============================================================

if len(points_3d) > 0:

    save_point_cloud(
        OUTPUT_FILE,
        points_3d
    )

    print()
    print("SUCCESS: 3D room point cloud created.")
    print()
    print(
        "Next file:"
    )
    print(
        os.path.abspath(OUTPUT_FILE)
    )

else:

    print()
    print("No 3D points were created.")
    print("Try moving the camera more slowly.")