import cv2
import numpy as np
import os
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

os.chdir('C:/coding/vscode_files/handTrackingProject') # 작업 디렉토리 설정

##############################Detection setup####################################
mp_hands = mp.solutions.hands     # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
#################################################################################



############################Setup Folders for Collection#########################
# Path for exported data, numpy arrays
DATA_PATH = os.path.join(os.getcwd(), 'MP_DATA') # cur path + 'MP_DATA'

# Actions that we try to detect
actions = np.array(['Hello', 'TV', 'Channel', 'Volume', 'On', 'Off', 'Next', 'Prev', 'OK', 'Light', 'Brightness'])                 # can add actions
# 
# Thirty videos worth of data
no_sequences = 400

# Videos are going to be 30 frames in length
sequence_length = 30

for action in actions: # making dir MP_DATA
    for sequence in range(1,no_sequences+1): 
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
#################################################################################


###############Preprocess Data and Create Labels and Features####################
# one-hot encoding
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10) # 95% of data will be used as train data
print(X.shape)
#################################################################################



# ######################Build and Train LSTM Neural Network########################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) # to debug training process

model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(30,126))) # 126 = 21(left hand)*3 + 21(right hand)*3
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback], batch_size=32) # change epochs to prevent overfitting

model.summary()    # for debug
# #################################################################################



##################################Save Weights####################################

model.save('action_test.h5')

model.load_weights('action_test.h5')
##############################optimizing model###################################
# convert h5 model into tflite model
h5model = tf.keras.models.load_model('action_test.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(h5model)
tflite_model = converter.convert()

with open('model.tflite','wb') as f:
    f.write(tflite_model)

################Evaluation using Confusion Matrix and Accuracy###################
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score  

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
#################################################################################


