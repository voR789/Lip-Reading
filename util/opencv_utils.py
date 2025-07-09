import cv2

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
