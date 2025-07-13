import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = True, 
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = .5
    )

def get_landmarks(image):
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks
    # return landmarks for each frame of video
    
def convert_landmarks_to_coordinates(landmarks, image_shape):
    h, w = image_shape[:2]
    return [(landmark.x*w, landmark.y*h) for landmark in landmarks.landmark]

def get_bounding_box(coords, image_shape, padding = 10):
    h, w = image_shape[:2]
    x, y = zip(*coords)
    x_min = max(min(x) - padding, 0)
    y_min = max(min(y) - padding, 0)
    x_max = min(max(x) + padding , w)
    y_max = min(max(y) + padding, h)
    return x_min, y_min, x_max, y_max

def landmark_dist(l1, l2):
    p1 = np.array(l1)
    p2 = np.array(l2)
    return np.linalg.norm(p2 - p1)

def area_of_points(arr, coords):
    # Gaussian Area Formula - "Shoelace Method"
    area = 0
    size = len(arr)
    for i in range(size - 1):
        area += coords[arr[i]][0]*coords[arr[i+1]][1]
        area -= coords[arr[i + 1]][0]*coords[arr[i]][1]
    area += coords[arr[-1]][0]*coords[arr[0]][1]
    area -= coords[arr[0]][0]*coords[arr[-1]][1]
    return area