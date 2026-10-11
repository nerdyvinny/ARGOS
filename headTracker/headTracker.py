import numpy as np
from math import radians, sin, cos, degrees, acos

# ==========================================
# CONFIGURATION
# ==========================================

# Lower = stricter
# 12 degrees works better than 15
ANGLE_THRESHOLD = 12

# Looking straight ahead
TARGET_VECTOR = np.array([0, 0, 1])

# ==========================================
# CONVERT HEAD POSE TO VECTOR
# ==========================================

def head_pose_to_vector(yaw, pitch):
    """
    Converts yaw and pitch angles into
    a normalized 3D direction vector.
    """

    yaw = radians(yaw)
    pitch = radians(pitch)

    x = sin(yaw) * cos(pitch)
    y = sin(pitch)
    z = cos(yaw) * cos(pitch)

    vector = np.array([x, y, z])

    return vector / np.linalg.norm(vector)

# ==========================================
# ANGLE BETWEEN TWO VECTORS
# ==========================================

def angle_between(v1, v2):

    dot = np.dot(v1, v2)

    dot = np.clip(
        dot / (np.linalg.norm(v1) * np.linalg.norm(v2)),
        -1.0,
        1.0
    )

    return degrees(acos(dot))

# ==========================================
# ATTENTION CHECK
# ==========================================

def is_target_reached(yaw, pitch):
    """
    Returns:
        reached (bool)
        error_angle (float)
    """

    current_vector = head_pose_to_vector(
        yaw,
        pitch
    )

    error_angle = angle_between(
        current_vector,
        TARGET_VECTOR
    )

    reached = error_angle <= ANGLE_THRESHOLD

    return reached, error_angle

# ==========================================
# TEST MODE
# ==========================================

if __name__ == "__main__":

    yaw = 0
    pitch = 0

    reached, error_angle = is_target_reached(
        yaw,
        pitch
    )

    print("HEAD TRACKER TEST")
    print("-----------------")
    print(f"Yaw: {yaw}")
    print(f"Pitch: {pitch}")
    print(f"Error Angle: {error_angle:.2f}°")

    if reached:
        print("✅ ATTENTIVE")
    else:
        print("⚠️ LOOKING AWAY")