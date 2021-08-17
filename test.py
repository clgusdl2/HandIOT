import numpy as np

ac = {'hello':1, 'bye':1, 'channel':2, 'next':3, 'prev':3}
actions = np.array(['OTya', 'TV', 'Channel', 'next', 'prev'])  
res = np.array([0.1, 0.5, 0.6, 0.2, 0.8])
actions[np.argmax(res)]
aclist = list(ac.keys())
aclist
