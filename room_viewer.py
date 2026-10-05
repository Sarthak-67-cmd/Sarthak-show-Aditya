import numpy as np
import cv2
import os
import math


# ============================================================
# SETTINGS
# ============================================================

WORLD_MAP_FILE = "room_world_map.npz"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750

POINT_LIMIT = 12000


# ============================================================
# LOAD WORLD MAP
# ============================================================

if not os.path.exists(WORLD_MAP_FILE):

    print()
    print("ERROR: room_world_map.npz was not found.")
    print()
    print("Run this first:")
    print()
    print("python room_geometry.py")
    print()

    input("Press Enter to exit...")
    raise SystemExit


data = np.load(
    WORLD_MAP_FILE
)

points = data["points"]

origin = data["origin"]

x_axis = data["x_axis"]
y_axis = data["y_axis"]
z_axis = data["z_axis"]


print()
print("========================================")
print("          ROOM 3D VIEWER")
print("========================================")
print()

print(
    f"Loaded {len(points)} points."
)

print()
print("Controls:")
print()
print("  W / S = Move forward / backward")
print("  A / D = Move left / right")
print("  Q / E = Move up / down")
print()
print("  Arrow Left / Right = Rotate")
print("  Arrow Up / Down    = Tilt")
print()
print("  Z / X = Zoom")
print()
print("  R = Reset camera")
print("  ESC = Exit")
print()


# ============================================================
# REDUCE POINT CLOUD
# ============================================================

if len(points) > POINT_LIMIT:

    indexes = np.linspace(
        0,
        len(points) - 1,
        POINT_LIMIT
    ).astype(int)

    points = points[indexes]


# ============================================================
# CAMERA
# ============================================================

viewer_position = np.array(
    [0.0, 1.5, -3.0],
    dtype=np.float64
)

yaw = 0.0
pitch = 0.0

zoom = 700.0


# ============================================================
# PROJECT 3D POINT
# ============================================================

def project_point(point):

    global viewer_position
    global yaw
    global pitch
    global zoom

    relative = (
        point -
        viewer_position
    )

    # --------------------------------------------------------
    # Yaw rotation
    # --------------------------------------------------------

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    x = (
        relative[0] * cos_yaw
        -
        relative[2] * sin_yaw
    )

    z = (
        relative[0] * sin_yaw
        +
        relative[2] * cos_yaw
    )

    y = relative[1]

    # --------------------------------------------------------
    # Pitch rotation
    # --------------------------------------------------------

    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    y2 = (
        y * cos_pitch
        -
        z * sin_pitch
    )

    z2 = (
        y * sin_pitch
        +
        z * cos_pitch
    )

    x2 = x

    # --------------------------------------------------------
    # Perspective
    # --------------------------------------------------------

    if z2 <= 0.05:

        return None

    screen_x = int(
        WINDOW_WIDTH / 2
        +
        (x2 / z2) * zoom
    )

    screen_y = int(
        WINDOW_HEIGHT / 2
        -
        (y2 / z2) * zoom
    )

    if (
        screen_x < -100
        or screen_x > WINDOW_WIDTH + 100
        or screen_y < -100
        or screen_y > WINDOW_HEIGHT + 100
    ):

        return None

    return (
        screen_x,
        screen_y,
        z2
    )


# ============================================================
# DRAW GRID
# ============================================================

def draw_grid(canvas):

    grid_size = 0.5

    grid_range = 10

    for x in np.arange(
        -grid_range,
        grid_range + grid_size,
        grid_size
    ):

        p1 = np.array(
            [x, 0, -grid_range]
        )

        p2 = np.array(
            [x, 0, grid_range]
        )

        a = project_point(p1)
        b = project_point(p2)

        if a is not None and b is not None:

            cv2.line(
                canvas,
                (a[0], a[1]),
                (b[0], b[1]),
                (45, 45, 45),
                1
            )

    for z in np.arange(
        -grid_range,
        grid_range + grid_size,
        grid_size
    ):

        p1 = np.array(
            [-grid_range, 0, z]
        )

        p2 = np.array(
            [grid_range, 0, z]
        )

        a = project_point(p1)
        b = project_point(p2)

        if a is not None and b is not None:

            cv2.line(
                canvas,
                (a[0], a[1]),
                (b[0], b[1]),
                (45, 45, 45),
                1
            )


# ============================================================
# DRAW AXES
# ============================================================

def draw_axes(canvas):

    axis_length = 2.0

    origin_point = np.array(
        [0.0, 0.0, 0.0]
    )

    x_end = np.array(
        [axis_length, 0.0, 0.0]
    )

    y_end = np.array(
        [0.0, axis_length, 0.0]
    )

    z_end = np.array(
        [0.0, 0.0, axis_length]
    )

    origin_screen = project_point(
        origin_point
    )

    x_screen = project_point(
        x_end
    )

    y_screen = project_point(
        y_end
    )

    z_screen = project_point(
        z_end
    )

    if origin_screen is None:
        return

    ox = origin_screen[0]
    oy = origin_screen[1]

    if x_screen is not None:

        cv2.line(
            canvas,
            (ox, oy),
            (x_screen[0], x_screen[1]),
            (0, 0, 255),
            3
        )

        cv2.putText(
            canvas,
            "X",
            (x_screen[0], x_screen[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if y_screen is not None:

        cv2.line(
            canvas,
            (ox, oy),
            (y_screen[0], y_screen[1]),
            (0, 255, 0),
            3
        )

        cv2.putText(
            canvas,
            "Y",
            (y_screen[0], y_screen[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if z_screen is not None:

        cv2.line(
            canvas,
            (ox, oy),
            (z_screen[0], z_screen[1]),
            (255, 0, 0),
            3
        )

        cv2.putText(
            canvas,
            "Z",
            (z_screen[0], z_screen[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


# ============================================================
# DRAW POINT CLOUD
# ============================================================

def draw_points(canvas):

    projected = []

    for point in points:

        result = project_point(
            point
        )

        if result is not None:

            sx, sy, depth = result

            projected.append(
                (
                    depth,
                    sx,
                    sy
                )
            )

    # Draw far points first
    projected.sort(
        reverse=True
    )

    for depth, sx, sy in projected:

        radius = 2

        cv2.circle(
            canvas,
            (sx, sy),
            radius,
            (100, 200, 255),
            -1
        )


# ============================================================
# DRAW CAMERA
# ============================================================

def draw_camera(canvas):

    camera_screen = project_point(
        viewer_position
    )

    if camera_screen is None:
        return

    cv2.circle(
        canvas,
        (
            camera_screen[0],
            camera_screen[1]
        ),
        7,
        (0, 0, 255),
        -1
    )


# ============================================================
# MAIN VIEWER LOOP
# ============================================================

while True:

    canvas = np.zeros(
        (
            WINDOW_HEIGHT,
            WINDOW_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Draw world
    # --------------------------------------------------------

    draw_grid(canvas)

    draw_points(canvas)

    draw_axes(canvas)

    draw_camera(canvas)

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "ROOM 3D VIEWER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"Points: {len(points)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        f"Camera: "
        f"{viewer_position[0]:.2f}, "
        f"{viewer_position[1]:.2f}, "
        f"{viewer_position[2]:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        "ESC = Exit",
        (20, WINDOW_HEIGHT - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "3D Room Viewer",
        canvas
    )

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if key == 27:
        break

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    movement_speed = 0.08

    forward = np.array(
        [
            math.sin(yaw),
            0,
            math.cos(yaw)
        ]
    )

    right = np.array(
        [
            math.cos(yaw),
            0,
            -math.sin(yaw)
        ]
    )

    if key == ord("w"):

        viewer_position += (
            forward *
            movement_speed
        )

    elif key == ord("s"):

        viewer_position -= (
            forward *
            movement_speed
        )

    elif key == ord("a"):

        viewer_position -= (
            right *
            movement_speed
        )

    elif key == ord("d"):

        viewer_position += (
            right *
            movement_speed
        )

    elif key == ord("q"):

        viewer_position[1] += movement_speed

    elif key == ord("e"):

        viewer_position[1] -= movement_speed

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    elif key == 81:  # Left arrow

        yaw -= 0.08

    elif key == 83:  # Right arrow

        yaw += 0.08

    elif key == 82:  # Up arrow

        pitch += 0.06

    elif key == 84:  # Down arrow

        pitch -= 0.06

    # --------------------------------------------------------
    # Zoom
    # --------------------------------------------------------

    elif key == ord("z"):

        zoom *= 1.1

    elif key == ord("x"):

        zoom *= 0.9

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    elif key == ord("r"):

        viewer_position = np.array(
            [0.0, 1.5, -3.0],
            dtype=np.float64
        )

        yaw = 0.0
        pitch = 0.0
        zoom = 700.0


cv2.destroyAllWindows()

print()
print("3D viewer closed.")