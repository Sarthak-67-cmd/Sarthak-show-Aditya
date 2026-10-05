import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import math


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "hand_landmarker.task"

CAMERA_INDEX = 0

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 800

PINCH_DISTANCE = 0.055

MOVE_SPEED = 3.0
ZOOM_SPEED = 1.5
ROTATE_SPEED = 2.0


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# HAND FUNCTIONS
# ============================================================

def distance_3d(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def get_hands(result):

    hands = []

    if not result.hand_landmarks:
        return hands

    for landmarks in result.hand_landmarks:

        thumb = landmarks[4]
        index = landmarks[8]

        pinch_distance = distance_3d(
            thumb,
            index
        )

        hands.append({
            "landmarks": landmarks,
            "pinch": pinch_distance < PINCH_DISTANCE,
            "x": index.x,
            "y": index.y,
            "pinch_distance": pinch_distance
        })

    return hands


# ============================================================
# CREATE REACTOR
# ============================================================

def create_reactor():

    parts = []

    # --------------------------------------------------------
    # CENTER CORE
    # --------------------------------------------------------

    core = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.55,
        height=0.25,
        resolution=64
    )

    core.compute_vertex_normals()

    core.paint_uniform_color(
        [0.1, 0.6, 1.0]
    )

    parts.append({
        "name": "CORE",
        "mesh": core
    })


    # --------------------------------------------------------
    # INNER RING
    # --------------------------------------------------------

    inner = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=0.72,
        tube_radius=0.08,
        radial_resolution=64,
        tubular_resolution=16
    )

    inner.compute_vertex_normals()

    inner.paint_uniform_color(
        [0.0, 0.8, 1.0]
    )

    parts.append({
        "name": "INNER RING",
        "mesh": inner
    })


    # --------------------------------------------------------
    # OUTER RING
    # --------------------------------------------------------

    outer = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=1.05,
        tube_radius=0.10,
        radial_resolution=64,
        tubular_resolution=16
    )

    outer.compute_vertex_normals()

    outer.paint_uniform_color(
        [0.1, 0.4, 1.0]
    )

    parts.append({
        "name": "OUTER RING",
        "mesh": outer
    })


    # --------------------------------------------------------
    # SIDE MODULES
    # --------------------------------------------------------

    modules = [
        ("RIGHT MODULE", [1.25, 0, 0]),
        ("LEFT MODULE", [-1.25, 0, 0]),
        ("TOP MODULE", [0, 1.25, 0]),
        ("BOTTOM MODULE", [0, -1.25, 0])
    ]


    for name, position in modules:

        box = o3d.geometry.TriangleMesh.create_box(
            width=0.28,
            height=0.28,
            depth=0.28
        )

        box.compute_vertex_normals()

        box.translate([
            position[0] - 0.14,
            position[1] - 0.14,
            position[2] - 0.14
        ])

        box.paint_uniform_color(
            [0.15, 0.55, 1.0]
        )

        parts.append({
            "name": name,
            "mesh": box
        })


    return parts


# ============================================================
# CREATE MODEL
# ============================================================

reactor_parts = create_reactor()


# ============================================================
# OPEN3D WINDOW
# ============================================================

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="TONY STARK - 3D HOLOGRAM",
    width=WINDOW_WIDTH,
    height=WINDOW_HEIGHT
)


# ============================================================
# ADD PARTS
# ============================================================

for part in reactor_parts:

    vis.add_geometry(
        part["mesh"]
    )


# ============================================================
# RENDER SETTINGS
# ============================================================

render_option = vis.get_render_option()

render_option.background_color = np.array([
    0.005,
    0.005,
    0.02
])

render_option.mesh_show_back_face = True

render_option.point_size = 3.0


# ============================================================
# CAMERA VIEW
# ============================================================

view = vis.get_view_control()

view.set_zoom(0.8)

view.set_front([
    0.0,
    0.0,
    -1.0
])

view.set_up([
    0.0,
    1.0,
    0.0
])


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    vis.destroy_window()

    landmarker.close()

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# VARIABLES
# ============================================================

selected_part = None

last_x = None
last_y = None

last_two_hand_distance = None
last_two_hand_angle = None

model_scale = 1.0

exploded = False

timestamp = 0


# ============================================================
# COLORS
# ============================================================

def update_colors():

    for i, part in enumerate(reactor_parts):

        if i == selected_part:

            part["mesh"].paint_uniform_color([
                0.0,
                1.0,
                1.0
            ])

        else:

            part["mesh"].paint_uniform_color([
                0.1,
                0.45,
                1.0
            ])

        vis.update_geometry(
            part["mesh"]
        )


# ============================================================
# SELECT PART
# ============================================================

def select_part(x, y):

    """
    Approximate hand-to-component selection.

    Webcam coordinates:

        left   -> left module
        right  -> right module
        top    -> top module
        bottom -> bottom module
        center -> core
    """

    if x < 0.35:

        return 4

    if x > 0.65:

        return 3

    if y < 0.35:

        return 5

    if y > 0.65:

        return 6

    return 0


# ============================================================
# MOVE PART
# ============================================================

def move_part(dx, dy):

    if selected_part is None:
        return

    part = reactor_parts[selected_part]

    part["mesh"].translate([
        dx * MOVE_SPEED,
        -dy * MOVE_SPEED,
        0
    ])

    vis.update_geometry(
        part["mesh"]
    )


# ============================================================
# SCALE MODEL
# ============================================================

def scale_model(factor):

    global model_scale

    new_scale = model_scale * factor

    new_scale = max(
        0.3,
        min(new_scale, 4.0)
    )

    actual_factor = (
        new_scale /
        model_scale
    )

    for part in reactor_parts:

        part["mesh"].scale(
            actual_factor,
            center=[0, 0, 0]
        )

        vis.update_geometry(
            part["mesh"]
        )

    model_scale = new_scale


# ============================================================
# ROTATE MODEL
# ============================================================

def rotate_model(angle):

    rotation = (
        o3d.geometry
        .get_rotation_matrix_from_xyz([
            0,
            angle,
            0
        ])
    )

    for part in reactor_parts:

        part["mesh"].rotate(
            rotation,
            center=[0, 0, 0]
        )

        vis.update_geometry(
            part["mesh"]
        )


# ============================================================
# EXPLODE
# ============================================================

def explode_model():

    global exploded

    if exploded:
        return

    exploded = True

    for i, part in enumerate(reactor_parts):

        if i == 0:
            continue

        center = np.array(
            part["mesh"].get_center()
        )

        direction = np.array([
            center[0],
            center[1],
            0.5
        ])

        length = np.linalg.norm(
            direction
        )

        if length > 0:

            direction /= length

        part["mesh"].translate(
            direction * 0.7
        )

        vis.update_geometry(
            part["mesh"]
        )


# ============================================================
# RESET
# ============================================================

def reset_model():

    global reactor_parts
    global selected_part
    global model_scale
    global exploded

    # Remove old objects

    for part in reactor_parts:

        vis.remove_geometry(
            part["mesh"],
            reset_bounding_box=False
        )


    # Create new reactor

    reactor_parts = create_reactor()


    for part in reactor_parts:

        vis.add_geometry(
            part["mesh"],
            reset_bounding_box=False
        )


    selected_part = None

    model_scale = 1.0

    exploded = False

    update_colors()


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):

    height, width = frame.shape[:2]

    for landmark in landmarks:

        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 255),
            -1
        )


    thumb = landmarks[4]

    index = landmarks[8]

    tx = int(
        thumb.x *
        width
    )

    ty = int(
        thumb.y *
        height
    )

    ix = int(
        index.x *
        width
    )

    iy = int(
        index.y *
        height
    )


    cv2.line(
        frame,
        (tx, ty),
        (ix, iy),
        (255, 255, 0),
        2
    )


    if distance_3d(
        thumb,
        index
    ) < PINCH_DISTANCE:

        cv2.circle(
            frame,
            (ix, iy),
            15,
            (0, 255, 0),
            3
        )


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(frame):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (560, 170),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )


    cv2.putText(
        frame,
        "TONY STARK 3D HOLOGRAM",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2
    )


    cv2.putText(
        frame,
        "PINCH = GRAB COMPONENT",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "TWO HANDS = ZOOM / ROTATE",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "E = EXPLODE",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "R = RESET     ESC = EXIT",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    if selected_part is not None:

        name = reactor_parts[
            selected_part
        ]["name"]

        cv2.putText(
            frame,
            "SELECTED: " + name,
            (width - 420, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


# ============================================================
# INITIAL COLORS
# ============================================================

update_colors()


# ============================================================
# START
# ============================================================

print()
print("========================================")
print("       TONY STARK 3D HOLOGRAM")
print("========================================")
print()
print("Controls:")
print()
print("  Pinch              = Select / Grab")
print("  Move hand          = Move component")
print("  Two hands apart    = Zoom in")
print("  Two hands together = Zoom out")
print("  Rotate two hands   = Rotate")
print("  E                  = Explode")
print("  R                  = Reset")
print("  ESC                = Exit")
print()
print("Starting...")
print()


# ============================================================
# MAIN LOOP
# ============================================================

running = True


while running:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read camera.")

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 33


    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    hands = get_hands(
        result
    )


    # --------------------------------------------------------
    # DRAW HANDS
    # --------------------------------------------------------

    if result.hand_landmarks:

        for landmarks in result.hand_landmarks:

            draw_hand(
                frame,
                landmarks
            )


    # ========================================================
    # ONE HAND
    # ========================================================

    if len(hands) == 1:

        hand = hands[0]


        if hand["pinch"]:

            x = hand["x"]
            y = hand["y"]


            # ------------------------------------------------
            # SELECT
            # ------------------------------------------------

            if selected_part is None:

                selected_part = select_part(
                    x,
                    y
                )

                update_colors()


            # ------------------------------------------------
            # MOVE
            # ------------------------------------------------

            if (
                last_x is not None
                and last_y is not None
            ):

                dx = x - last_x
                dy = y - last_y

                move_part(
                    dx,
                    dy
                )


            last_x = x
            last_y = y


        else:

            last_x = None
            last_y = None

            selected_part = None

            update_colors()


    # ========================================================
    # TWO HANDS
    # ========================================================

    elif len(hands) >= 2:

        h1 = hands[0]
        h2 = hands[1]


        x1 = h1["x"]
        y1 = h1["y"]

        x2 = h2["x"]
        y2 = h2["y"]


        # ----------------------------------------------------
        # HAND DISTANCE
        # ----------------------------------------------------

        current_distance = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )


        if last_two_hand_distance is not None:

            difference = (
                current_distance -
                last_two_hand_distance
            )


            if abs(difference) > 0.005:

                factor = (
                    1.0 +
                    difference *
                    ZOOM_SPEED
                )


                factor = max(
                    0.95,
                    min(
                        factor,
                        1.05
                    )
                )


                scale_model(
                    factor
                )


        last_two_hand_distance = (
            current_distance
        )


        # ----------------------------------------------------
        # ROTATION
        # ----------------------------------------------------

        current_angle = math.atan2(
            y2 - y1,
            x2 - x1
        )


        if last_two_hand_angle is not None:

            angle_difference = (
                current_angle -
                last_two_hand_angle
            )


            # Handle angle wraparound

            if angle_difference > math.pi:

                angle_difference -= (
                    2 * math.pi
                )


            if angle_difference < -math.pi:

                angle_difference += (
                    2 * math.pi
                )


            if abs(angle_difference) > 0.005:

                rotate_model(
                    angle_difference *
                    ROTATE_SPEED
                )


        last_two_hand_angle = (
            current_angle
        )


    # ========================================================
    # NO HANDS
    # ========================================================

    else:

        last_x = None
        last_y = None

        last_two_hand_distance = None
        last_two_hand_angle = None


    # ========================================================
    # UPDATE OPEN3D
    # ========================================================

    vis.poll_events()

    vis.update_renderer()


    # ========================================================
    # HUD
    # ========================================================

    draw_hud(
        frame
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "HAND TRACKING",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        running = False


    elif key == ord("e"):

        print("Exploding model...")

        explode_model()


    elif key == ord("r"):

        print("Resetting model...")

        reset_model()


# ============================================================
# CLEANUP
# ============================================================

print()
print("Stopping...")


cap.release()

cv2.destroyAllWindows()

landmarker.close()

vis.destroy_window()


print("Hologram stopped.")
print()