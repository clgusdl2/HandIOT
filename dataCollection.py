import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
os.chdir('C:/coding/vscode_files/handTrackingProject') # dir set


### detection setup
mp_hands = mp.solutions.hands     # Hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    rh = np.zeros(21*3)
    lh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            for _, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                if handedness_dict['classification'][0]['label'] == 'Left':
                    rh = np.zeros(21*3)
                    lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten() # each landmark has 3 factors(x, y, z)            
                else:
                    rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
                    lh = np.zeros(21*3)
        elif len(results.multi_hand_landmarks) == 2:
            rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
            lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[1].landmark)]).flatten()
    return np.concatenate([rh, lh])



### setup folders for collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), 'MP_DATA') # cur path + 'MP_DATA'

# Actions that we try to detect
actions = np.array(['Hello'])                 # can add actions, change into other word to collect data one at once
## 'Hello', 'TV', 'Channel', 'Volume', 'On', 'Off', 'Next', 'Prev', 'OK'

# length of data
no_sequences = 400  # 400 data folders will be created

# Videos are going to be 30 frames in length
sequence_length = 30

# could stack data
dataLen = len(list(os.listdir('C:/coding/vscode_files/handTrackingProject/MP_DATA/Brightness')))

for action in actions: # making dir MP_DATA
    for sequence in range(1,no_sequences+1): 
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence+dataLen)))
        except:
            pass



### Collect Keypoint Values for Training and Testing
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(False,2, 0.6, 0.6) as hands:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(1, no_sequences+1):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                if results.multi_hand_landmarks:
                # print(results.multi_hand_landmarks)
                    for handLms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence+dataLen), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()
#################################################################################
