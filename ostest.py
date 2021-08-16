import os
import numpy as np
os.chdir('C:/coding/vscode_files/handTrackingProject')



DATA_PATH = os.path.join(os.getcwd(), 'MP_DATA')

# Actions that we try to detect
actions = np.array(['TV', 'On', 'Off'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

for action in actions: 
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass