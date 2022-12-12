from symbol import parameters
import torch
import pickle
import os
# model = torch.load("../ultra_x2.pth")
# for k, v in model.items():
#     print(k, v)
# with open('./ultra_x2.txt', 'w') as file:
#     file.write(pickle.dumps(model))

path = '/home/ubuntu/data/main/tableSR/frames/nemo/Productreview/540p/videoplayback1'
filenames = sorted(os.listdir(path))
print(filenames)