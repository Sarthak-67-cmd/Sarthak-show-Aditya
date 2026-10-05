import numpy as np
import os
from scipy.spatial import cKDTree


# ============================================================
# SETTINGS
# ============================================================

POINT_CLOUD_FILE = "room_pointcloud.ply"

MIN_POINTS = 100

# RANSAC settings for floor detection
RANSAC_ITERATIONS = 500
FLOOR_DISTANCE_THRESHOLD = 0.08

# Ignore extremely distant points
MAX_DISTANCE = 20.0


# ============================================================
# LOAD PLY
# ============================================================

def load_ply(filename):

    if not os.path.exists(filename):

        print()
        print("ERROR:")
        print(f"Could not find {filename}")
        print()
        print("Run room_3d.py first to create the point cloud.")
        return None

    points = []

    try:

        with open(filename, "r", encoding="utf-8") as file:

            header_finished = False

            vertex_count = 0

            for line in file:

                line = line.strip()

                if line == "end_header":

                    header_finished = True
                    continue

                if not header_finished:

                    if line.startswith("element vertex"):

                        vertex_count = int(
                            line.split()[-1]
                        )

                    continue

                parts = line.split()

                if len(parts) < 3:
                    continue

                try:

                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])

                    if not np.isfinite(x):
                        continue

                    if not np.isfinite(y):
                        continue

                    if not np.isfinite(z):
                        continue

                    points.append(
                        [x, y, z]
                    )

                except ValueError:

                    continue

        points = np.asarray(
            points,
            dtype=np.float64
        )

        print()
        print("========================================")
        print("          POINT CLOUD LOADED")
        print("========================================")
        print()
        print(f"Header vertices: {vertex_count}")
        print(f"Valid points:    {len(points)}")
        print()

        return points

    except Exception as error:

        print()
        print("ERROR loading point cloud:")
        print(error)

        return None


# ============================================================
# CLEAN POINT CLOUD
# ============================================================

def clean_points(points):

    if points is None:
        return None

    distances = np.linalg.norm(
        points,
        axis=1
    )

    mask = (
        np.isfinite(distances)
        &
        (distances < MAX_DISTANCE)
    )

    cleaned = points[mask]

    print(
        f"Points after cleaning: {len(cleaned)}"
    )

    return cleaned


# ============================================================
# FIT PLANE
# ============================================================

def plane_from_points(p1, p2, p3):

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(
        v1,
        v2
    )

    length = np.linalg.norm(normal)

    if length < 1e-8:

        return None

    normal = normal / length

    d = -np.dot(
        normal,
        p1
    )

    return normal, d


# ============================================================
# DISTANCE TO PLANE
# ============================================================

def plane_distances(points, normal, d):

    return np.abs(
        points @ normal + d
    )


# ============================================================
# FIND FLOOR
# ============================================================

def find_floor(points):

    if len(points) < MIN_POINTS:

        print("Not enough points for floor detection.")
        return None

    print()
    print("Detecting floor...")
    print(
        f"RANSAC iterations: {RANSAC_ITERATIONS}"
    )

    best_normal = None
    best_d = None
    best_count = 0
    best_mask = None

    rng = np.random.default_rng(42)

    for iteration in range(RANSAC_ITERATIONS):

        indexes = rng.choice(
            len(points),
            3,
            replace=False
        )

        p1 = points[indexes[0]]
        p2 = points[indexes[1]]
        p3 = points[indexes[2]]

        result = plane_from_points(
            p1,
            p2,
            p3
        )

        if result is None:
            continue

        normal, d = result

        # ----------------------------------------------------
        # Floor should be approximately horizontal.
        #
        # We assume Y is the vertical axis.
        # ----------------------------------------------------

        vertical_alignment = abs(
            np.dot(
                normal,
                np.array([0.0, 1.0, 0.0])
            )
        )

        # Reject planes that are too vertical.
        if vertical_alignment < 0.70:
            continue

        distances = plane_distances(
            points,
            normal,
            d
        )

        mask = (
            distances <
            FLOOR_DISTANCE_THRESHOLD
        )

        count = np.sum(mask)

        if count > best_count:

            best_count = count

            best_normal = normal.copy()

            best_d = d

            best_mask = mask.copy()

    if best_normal is None:

        print()
        print("Could not detect a floor plane.")
        return None

    # --------------------------------------------------------
    # Make normal point upward
    # --------------------------------------------------------

    if best_normal[1] < 0:

        best_normal = -best_normal
        best_d = -best_d

    floor_points = points[best_mask]

    print()
    print("========================================")
    print("             FLOOR FOUND")
    print("========================================")
    print()

    print(
        f"Floor points: {len(floor_points)}"
    )

    print(
        f"Floor normal: "
        f"{best_normal[0]:.4f}, "
        f"{best_normal[1]:.4f}, "
        f"{best_normal[2]:.4f}"
    )

    print(
        f"Plane equation:"
    )

    print(
        f"{best_normal[0]:.4f}x + "
        f"{best_normal[1]:.4f}y + "
        f"{best_normal[2]:.4f}z + "
        f"{best_d:.4f} = 0"
    )

    return (
        best_normal,
        best_d,
        floor_points
    )


# ============================================================
# CREATE ROOM COORDINATE SYSTEM
# ============================================================

def create_world_coordinates(
    points,
    floor_normal,
    floor_d
):

    # --------------------------------------------------------
    # Find a representative floor origin
    # --------------------------------------------------------

    floor_distances = np.abs(
        points @ floor_normal + floor_d
    )

    floor_mask = (
        floor_distances <
        FLOOR_DISTANCE_THRESHOLD
    )

    floor_points = points[floor_mask]

    if len(floor_points) == 0:

        print("Could not create world origin.")
        return None

    origin = np.mean(
        floor_points,
        axis=0
    )

    # --------------------------------------------------------
    # Force origin onto floor plane
    # --------------------------------------------------------

    correction = (
        np.dot(
            floor_normal,
            origin
        )
        + floor_d
    )

    origin = (
        origin
        - correction * floor_normal
    )

    # --------------------------------------------------------
    # World Y axis
    # --------------------------------------------------------

    world_y = floor_normal.copy()

    world_y = (
        world_y /
        np.linalg.norm(world_y)
    )

    # --------------------------------------------------------
    # Choose world X direction
    #
    # Use the direction in the point cloud with the
    # largest horizontal spread.
    # --------------------------------------------------------

    centered = (
        floor_points
        - origin
    )

    horizontal = centered.copy()

    horizontal[:, 1] = 0

    if len(horizontal) > 10:

        covariance = np.cov(
            horizontal[:, [0, 2]].T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        direction = eigenvectors[
            :, np.argmax(eigenvalues)
        ]

        world_x = np.array(
            [
                direction[0],
                0.0,
                direction[1]
            ],
            dtype=np.float64
        )

    else:

        world_x = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64
        )

    # Normalize
    world_x = (
        world_x /
        np.linalg.norm(world_x)
    )

    # --------------------------------------------------------
    # World Z
    # --------------------------------------------------------

    world_z = np.cross(
        world_x,
        world_y
    )

    world_z = (
        world_z /
        np.linalg.norm(world_z)
    )

    # Recalculate X to ensure orthogonality
    world_x = np.cross(
        world_y,
        world_z
    )

    world_x = (
        world_x /
        np.linalg.norm(world_x)
    )

    return {
        "origin": origin,
        "x_axis": world_x,
        "y_axis": world_y,
        "z_axis": world_z
    }


# ============================================================
# CONVERT POINTS TO WORLD SPACE
# ============================================================

def transform_points(
    points,
    coordinate_system
):

    origin = coordinate_system["origin"]

    x_axis = coordinate_system["x_axis"]
    y_axis = coordinate_system["y_axis"]
    z_axis = coordinate_system["z_axis"]

    centered = points - origin

    world_x = centered @ x_axis
    world_y = centered @ y_axis
    world_z = centered @ z_axis

    transformed = np.column_stack(
        (
            world_x,
            world_y,
            world_z
        )
    )

    return transformed


# ============================================================
# SAVE WORLD MAP
# ============================================================

def save_world_map(
    filename,
    points,
    coordinate_system
):

    try:

        np.savez(
            filename,
            points=points,
            origin=coordinate_system["origin"],
            x_axis=coordinate_system["x_axis"],
            y_axis=coordinate_system["y_axis"],
            z_axis=coordinate_system["z_axis"]
        )

        print()
        print("========================================")
        print("        WORLD MAP SAVED")
        print("========================================")
        print()

        print(
            f"File: {os.path.abspath(filename)}"
        )

        print(
            f"Points: {len(points)}"
        )

        print()

        return True

    except Exception as error:

        print("ERROR saving world map:")
        print(error)

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("       ROOM GEOMETRY PROCESSOR")
    print("========================================")
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    points = load_ply(
        POINT_CLOUD_FILE
    )

    if points is None:
        return

    if len(points) < MIN_POINTS:

        print(
            "Not enough points."
        )

        return

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    points = clean_points(
        points
    )

    # --------------------------------------------------------
    # Floor
    # --------------------------------------------------------

    result = find_floor(
        points
    )

    if result is None:
        return

    floor_normal, floor_d, floor_points = result

    # --------------------------------------------------------
    # World coordinates
    # --------------------------------------------------------

    coordinate_system = create_world_coordinates(
        points,
        floor_normal,
        floor_d
    )

    if coordinate_system is None:
        return

    print()
    print("========================================")
    print("       ROOM COORDINATE SYSTEM")
    print("========================================")
    print()

    print(
        "Origin:"
    )

    print(
        coordinate_system["origin"]
    )

    print()

    print(
        "X axis:"
    )

    print(
        coordinate_system["x_axis"]
    )

    print()

    print(
        "Y axis:"
    )

    print(
        coordinate_system["y_axis"]
    )

    print()

    print(
        "Z axis:"
    )

    print(
        coordinate_system["z_axis"]
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    world_points = transform_points(
        points,
        coordinate_system
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_world_map(
        "room_world_map.npz",
        world_points,
        coordinate_system
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("========================================")
    print("           ROOM STATISTICS")
    print("========================================")
    print()

    min_values = np.min(
        world_points,
        axis=0
    )

    max_values = np.max(
        world_points,
        axis=0
    )

    size = (
        max_values -
        min_values
    )

    print(
        f"X range: {min_values[0]:.3f} "
        f"to {max_values[0]:.3f}"
    )

    print(
        f"Y range: {min_values[1]:.3f} "
        f"to {max_values[1]:.3f}"
    )

    print(
        f"Z range: {min_values[2]:.3f} "
        f"to {max_values[2]:.3f}"
    )

    print()

    print(
        f"Approx. width:  {size[0]:.3f}"
    )

    print(
        f"Approx. height: {size[1]:.3f}"
    )

    print(
        f"Approx. depth:  {size[2]:.3f}"
    )

    print()

    print("World map is ready.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()