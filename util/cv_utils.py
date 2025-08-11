import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = True, 
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = .5
    )

def load_video(path):
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        returned, frame = capture.read()
        if not returned:
            break
            # break when theres no more output
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        # convert each frame to RGB, as openCV is BGR
    
    capture.release()
    return frames

def get_landmarks(image):
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks
    # return landmarks for each frame of video
    
def convert_landmarks_to_coordinates(landmarks, image_shape):
    h, w = image_shape[:2]
    return [(landmark.x*w, landmark.y*h) for landmark in landmarks.landmark]

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

def normalize_vertical_measurements(coords, measurements):
    face_height = landmark_dist(coords[10], coords[152])
    return [m/face_height for m in measurements]

def normalize_horizontal_measurements(coords, measurements):
    eye_dist = landmark_dist(coords[133], coords[362])
    return [m/eye_dist for m in measurements]

def normalize_area(coords, measurements):
    face_height = landmark_dist(coords[10], coords[152])
    eye_dist = landmark_dist(coords[133], coords[362])
    return [m/(face_height*eye_dist) for m in measurements]

def khz_to_frame(start, end, fps=25, khz=25000):
    return round((int(start) / khz) * fps) , round((int(end) / khz) * fps)
    
def extractClips(align_path):
    timeSegments = []
    with open(align_path, 'r') as f:
        lines = f.readlines();
    for line in lines[1:-1]: # Skip "sil"
        start, end, word = line.strip().split()
        if word == 'sil':
            continue
        start_f, end_f = khz_to_frame(start,end)
        timeSegments.append((start_f, end_f, word))
    return timeSegments

def setup_landmarks(frame):
    # Crop and proccess landmarks
    landmarks = get_landmarks(frame)[0] # 0 being the first face found
    coords = convert_landmarks_to_coordinates(landmarks, frame.shape)
    return coords

# old with list of dictionaries, each dic key being a column header 
# def fetch_data(frame, frame_number, previous_data):    
#     # Primary, normalized measurements
#     coords = setup_landmarks(frame)
    
#     upper_vermillion_height = landmark_dist(coords[13], coords[0])
#     lower_verillion_height = landmark_dist(coords[14], coords[17])
#     vert_distance = landmark_dist(coords[13], coords[14])
#     hori_distance = landmark_dist(coords[308], coords[78])
#     lower_jaw_distance = landmark_dist(coords[14], coords[152])
#     inner_mouth_area = abs(area_of_points([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],coords))

#     # Normalized to face size
#     norm_uvh, norm_lvh, norm_vd, norm_jd = normalize_vertical_measurements(coords, [upper_vermillion_height,lower_verillion_height,vert_distance,lower_jaw_distance])
#     norm_hd, = normalize_horizontal_measurements(coords, [hori_distance])
#     norm_ima, = normalize_area(coords, [inner_mouth_area])
    
#     # Individual Landmark Velocities and landmarks 2 Dimensional
#     # Normalize coordingates first to consistent origin - midpt of eyes
#     eye_midpt = (((coords[133][0] + coords[362][0])/2), ((coords[133][1] + coords[362][1])/2))
#     norm_coords = [((coord[0]-eye_midpt[0]),(coord[1]-eye_midpt[1])) for coord in coords]
#     key_landmarks_index = [78, 308, 13, 14, 152, 311, 81, 402, 178] 
    
#     key_norm_coords = {}
#     for i in range(len(key_landmarks_index)):
#         x = norm_coords[key_landmarks_index[i]][0]
#         y = norm_coords[key_landmarks_index[i]][1]
#         key_norm_coords[f"landmark_{key_landmarks_index[i]}_x"] = x
#         key_norm_coords[f"landmark_{key_landmarks_index[i]}_y"] = y
    
#     # key_norm_coords = {f"landmark_{i}":norm_coords[i] for i in key_landmarks_index}
    
#     velocities = {}
#     for i in range(len(key_landmarks_index)):
#         if(frame_number != 0):
#             v_x = norm_coords[key_landmarks_index[i]][0] - previous_data[1][f"landmark_{key_landmarks_index[i]}_x"]
#             v_y = norm_coords[key_landmarks_index[i]][1] - previous_data[1][f"landmark_{key_landmarks_index[i]}_y"]
#         else:
#             v_x = 0
#             v_y = 0
#         velocities[f"landmark_{key_landmarks_index[i]}_x"] = v_x
#         velocities[f"landmark_{key_landmarks_index[i]}_y"] = v_y
            
#         # for i in key_landmarks_index:
#         #     v_x = (norm_coords[i][0] - previous_data[3][i][0])
#         #     v_y = (norm_coords[i][1] - previous_data[3][i][1])  
#         #     velocities[f"landmark_{i}"] = (v_x,v_y)
    
#     # Engineered Feature Velocities - 1 Dimensional
#     if frame_number == 0:
#         norm_vert_opening_velocity = 0
#         norm_hori_opening_velocity = 0
#     else:
#         norm_vert_opening_velocity = norm_vd - previous_data[0]['norm_vd']
#         norm_hori_opening_velocity = norm_hd - previous_data[0]['norm_hd'] # lip corner velocity 
    
#     engineered_features = {
#         'norm_uvh': norm_uvh,
#         'norm_lvh': norm_lvh,
#         'norm_vd': norm_vd,
#         'norm_hd': norm_hd,
#         'norm_jd': norm_jd,
#         'norm_ima': norm_ima,
#         'norm_vert_opening_velocity': norm_vert_opening_velocity,
#         'norm_hori_opening_velocity': norm_hori_opening_velocity,
#     }
    
#     # return 3 dictionaries (used in ML) and one list (used for calcs)
#     return [engineered_features, key_norm_coords, velocities, norm_coords]
    

# Added: Accelerations
def fetch_data(frame, frame_number, previous_data):    
    # Primary, normalized measurements
    coords = setup_landmarks(frame)
    
    # Engineered Features
    upper_vermillion_height = landmark_dist(coords[13], coords[0])
    lower_verillion_height = landmark_dist(coords[14], coords[17])
    vert_distance = landmark_dist(coords[13], coords[14])
    hori_distance = landmark_dist(coords[308], coords[78])
    lower_jaw_distance = landmark_dist(coords[13], coords[152]) # Changed to landmark 13 (upper mouth to jaw)
    inner_mouth_area = abs(area_of_points([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],coords))
    # Normalized to face size
    norm_uvh, norm_lvh, norm_vd, norm_jd = normalize_vertical_measurements(coords, [upper_vermillion_height,lower_verillion_height,vert_distance,lower_jaw_distance])
    norm_hd, = normalize_horizontal_measurements(coords, [hori_distance])
    norm_ima, = normalize_area(coords, [inner_mouth_area])
    # Engineered Feature Velocities - 1 Dimensional
    if frame_number == 0:
        norm_vert_opening_velocity = 0
        norm_hori_opening_velocity = 0
    else:
        norm_vert_opening_velocity = norm_vd - previous_data[0][2]
        norm_hori_opening_velocity = norm_hd - previous_data[0][3] # lip corner velocity 
    engineered_features = [norm_uvh, norm_lvh, norm_vd, norm_hd, norm_jd, norm_ima, norm_vert_opening_velocity, norm_hori_opening_velocity]
    
    # Coordinates
    # Normalize coordingates first to consistent origin - midpt of eyes
    eye_midpt = (((coords[133][0] + coords[362][0])/2), ((coords[133][1] + coords[362][1])/2))
    norm_coords = [((coord[0]-eye_midpt[0]),(coord[1]-eye_midpt[1])) for coord in coords]
    key_landmarks_index = [78, 308, 13, 14, 152, 311, 81, 402, 178] 
    
    key_norm_coords = []
    for i in range(len(key_landmarks_index)):
        x = norm_coords[key_landmarks_index[i]][0]
        y = norm_coords[key_landmarks_index[i]][1]
        key_norm_coords.extend([x, y])
    
    # Velocities
    velocities = []
    for i in range(len(key_landmarks_index)):
        if(frame_number != 0): # Pull relavant coords out of norm_coords, and convert to velocities with last set of data
            v_x = norm_coords[key_landmarks_index[i]][0] - previous_data[1][i*2]
            v_y = norm_coords[key_landmarks_index[i]][1] - previous_data[1][i*2 + 1]
        else:
            v_x = 0
            v_y = 0
        velocities.extend([v_x, v_y])
    
    # Accelerations
    accelerations = []
    for i in range(0, len(velocities), 2):
        if(frame_number!= 0):
            a_x = velocities[i] - previous_data[2][i]
            a_y = velocities[i + 1] - previous_data[2][i + 1]
        else:
            a_x = 0
            a_y = 0
        accelerations.extend([a_x, a_y])
    
    # return 4 2d data matrices for each frame
    return [engineered_features, key_norm_coords, velocities, accelerations]


# clip is a cropped list of frames
def proccess_Clip(clip):
    index = 0
    data = []
    engineered_features = []
    coordinates = []
    velocities = []
    accelerations = []
    for frame in clip:
        data = fetch_data(frame, index, data) # inputs the last data fetched 
        index += 1
        engineered_features.append(data[0])
        coordinates.append(data[1])
        velocities.append(data[2])
        accelerations.append(data[3])
    return engineered_features, coordinates, velocities, accelerations

def collate_fn(batch):
    # clean any corrupted data first\
    cleaned = []
    for sample in batch:
        x1, x2, x3, x4, y = sample
        if x1.shape[0] == 0 or x2.shape[0] == 0 or x3.shape[0] == 0 or x4.shape[0] == 0:
            # skip
            continue
        else:
            cleaned.append(sample)
    
    x1, x2, x3, x4, y = zip(*cleaned) # unpack getItem data
    
    
    
    x1_padded = pad_sequence(x1, batch_first=True) # outputs 3dim tensor of (batch_size, max_seq_len, feature_size) "True Tensors"
    x2_padded = pad_sequence(x2, batch_first=True)
    x3_padded = pad_sequence(x3, batch_first=True)
    x4_padded = pad_sequence(x4, batch_first=True)
    
    y_tensor = torch.stack(y)  # since labels are already tensors (dtype long)

    return x1_padded, x2_padded, x3_padded, x4_padded, y_tensor


