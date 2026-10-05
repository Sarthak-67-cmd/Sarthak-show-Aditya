import cv2
import numpy as np
import math
import os


# ============================================================
# SETTINGS
# ============================================================

WORLD_MAP_FILE = "room_world_map.npz"

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750

CUBE_SIZE = 0.5

POINT_LIMIT = 6000


# ============================================================
# CHECK WORLD MAP
# ============================================================

if not os.path.exists(WORLD_MAP_FILE):

    print("ERROR: room_world_map.npz not found.")
    print()
    print("Run:")
    print("python room_geometry.py")
    input()
    raise SystemExit


# ============================================================
# LOAD ROOM
# ============================================================

data = np.load(WORLD_MAP_FILE)

room_points = data["points"]


if len(room_points) > POINT_LIMIT:

    indexes = np.linspace(
        0,
        len(room_points) - 1,
        POINT_LIMIT
    ).astype(int)

    room_points = room_points[indexes]


print()
print("========================================")
print("       HOLOGRAPHIC OBJECT TEST")
print("========================================")
print()
print(f"Room points: {len(room_points)}")
print()
print("Controls:")
print()
print("W/S = Forward / Back")
print("A/D = Left / Right")
print("Q/E = Up / Down")
print()
print("Arrow keys = Look around")
print("Z/X = Zoom")
print()
print("R = Reset")
print("ESC = Exit")
print()


# ============================================================
# VIEWER CAMERA
# ============================================================

camera_position = np.array(
    [0.0, 1.5, -3.0],
    dtype=np.float64
)

yaw = 0.0
pitch = 0.0

zoom = 700.0


# ============================================================
# HOLOGRAM OBJECT
# ============================================================

cube_position = np.array(
    [0.0, 1.0, 2.0],
    dtype=np.float64
)

cube_size = CUBE_SIZE


# ============================================================
# PROJECT 3D → SCREEN
# ============================================================

def project(point):

    relative = (
        point -
        camera_position
    )

    # --------------------------------------------------------
    # YAW
    # --------------------------------------------------------

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    x = (
        relative[0] * cy
        -
        relative[2] * sy
    )

    z = (
        relative[0] * sy
        +
        relative[2] * cy
    )

    y = relative[1]

    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    y2 = (
        y * cp
        -
        z * sp
    )

    z2 = (
        y * sp
        +
        z * cp
    )

    if z2 <= 0.05:

        return None

    sx = int(
        WINDOW_WIDTH / 2
        +
        x / z2 * zoom
    )

    sy_screen = int(
        WINDOW_HEIGHT / 2
        -
        y2 / z2 * zoom
    )

    return (
        sx,
        sy_screen,
        z2
    )


# ============================================================
# CUBE VERTICES
# ============================================================

def get_cube_vertices():

    s = cube_size / 2

    cx = cube_position[0]
    cy = cube_position[1]
    cz = cube_position[2]

    return [

        np.array([cx - s, cy - s, cz - s]),
        np.array([cx + s, cy - s, cz - s]),
        np.array([cx + s, cy + s, cz - s]),
        np.array([cx - s, cy + s, cz - s]),

        np.array([cx - s, cy - s, cz + s]),
        np.array([cx + s, cy - s, cz + s]),
        np.array([cx + s, cy + s, cz + s]),
        np.array([cx - s, cy + s, cz + s]),

    ]


# ============================================================
# DRAW CUBE
# ============================================================

def draw_cube(canvas):

    vertices = get_cube_vertices()

    projected = []

    for vertex in vertices:

        p = project(vertex)

        if p is None:

            return

        projected.append(p)

    edges = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),

    ]

    for a, b in edges:

        p1 = projected[a]
        p2 = projected[b]

        cv2.line(
            canvas,
            (p1[0], p1[1]),
            (p2[0], p2[1]),
            (255, 255, 255),
            3
        )

    # --------------------------------------------------------
    # Cube center
    # --------------------------------------------------------

    center = project(
        cube_position
    )

    if center is not None:

        cv2.circle(
            canvas,
            (center[0], center[1]),
            7,
            (255, 255, 255),
            -1
        )


# ============================================================
# DRAW ROOM POINTS
# ============================================================

def draw_room(canvas):

    for point in room_points:

        p = project(point)

        if p is None:
            continue

        x = p[0]
        y = p[1]

        if (
            x < 0
            or x >= WINDOW_WIDTH
            or y < 0
            or y >= WINDOW_HEIGHT
        ):
            continue

        cv2.circle(
            canvas,
            (x, y),
            1,
            (120, 120, 120),
            -1
        )


# ============================================================
# DRAW FLOOR GRID
# ============================================================

def draw_grid(canvas):

    for x in np.arange(
        -5,
        5.1,
        0.5
    ):

        p1 = project(
            np.array(
                [x, 0, -5]
            )
        )

        p2 = project(
            np.array(
                [x, 0, 5]
            )
        )

        if p1 is not None and p2 is not None:

            cv2.line(
                canvas,
                (p1[0], p1[1]),
                (p2[0], p2[1]),
                (40, 40, 40),
                1
            )

    for z in np.arange(
        -5,
        5.1,
        0.5
    ):

        p1 = project(
            np.array(
                [-5, 0, z]
            )
        )

        p2 = project(
            np.array(
                [5, 0, z]
            )
        )

        if p1 is not None and p2 is not None:

            cv2.line(
                canvas,
                (p1[0], p1[1]),
                (p2[0], p2[1]),
                (40, 40, 40),
                1
            )


# ============================================================
# MAIN LOOP
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
    # ROOM
    # --------------------------------------------------------

    draw_grid(canvas)

    draw_room(canvas)

    # --------------------------------------------------------
    # HOLOGRAM
    # --------------------------------------------------------

    draw_cube(canvas)

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "HOLOGRAPHIC OBJECT",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "Virtual Cube",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    cv2.putText(
        canvas,
        f"Object: "
        f"{cube_position[0]:.2f}, "
        f"{cube_position[1]:.2f}, "
        f"{cube_position[2]:.2f}",
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
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "Holographic Object",
        canvas
    )

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if key == 27:
        break

    # --------------------------------------------------------
    # MOVEMENT
    # --------------------------------------------------------

    speed = 0.08

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

        camera_position += (
            forward * speed
        )

    elif key == ord("s"):

        camera_position -= (
            forward * speed
        )

    elif key == ord("a"):

        camera_position -= (
            right * speed
        )

    elif key == ord("d"):

        camera_position += (
            right * speed
        )

    elif key == ord("q"):

        camera_position[1] += speed

    elif key == ord("e"):

        camera_position[1] -= speed

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    elif key == 81:

        yaw -= 0.08

    elif key == 83:

        yaw += 0.08

    elif key == 82:

        pitch += 0.06

    elif key == 84:

        pitch -= 0.06

    # --------------------------------------------------------
    # ZOOM
    # --------------------------------------------------------

    elif key == ord("z"):

        zoom *= 1.1

    elif key == ord("x"):

        zoom *= 0.9

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    elif key == ord("r"):

        camera_position = np.array(
            [0.0, 1.5, -3.0],
            dtype=np.float64
        )

        yaw = 0.0
        pitch = 0.0
        zoom = 700.0


cv2.destroyAllWindows()

print()
print("Holographic object viewer closed.")