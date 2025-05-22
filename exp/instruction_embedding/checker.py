import pickle
import numpy as np

path = " "
with open(path, 'rb') as f:
    data = pickle.load(f)
    
    for k, i in data.items():
        print(k)
        print(i.shape)
        print("="*50)
        break
