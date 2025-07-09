import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False, max_num_faces = 1)

def get_landmarks(image):
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0]
    # return lankmarks for each frame of video