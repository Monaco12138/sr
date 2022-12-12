from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


def imageRead( path ):
    #  return transforms.ToTensor()(
    #      Image.open(path).convert('RGB')
    #  )
    return np.array( Image.open(path).convert('RGB') )

def cv2Read( path ):
    return cv2.imread( path )

def calc_psnr(sr, hr, scale=1, rgb_range=1):

    valid = (sr - hr) / rgb_range
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

srPath = '/home/ubuntu/data/main/invertible/DIV2KFW1-800/0001_FW.png'
hrPath = '/home/ubuntu/data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X2_1-800/0001x2.png'


print( psnr( cv2Read(srPath), cv2Read('./test.png'), data_range=255) ) 

print( psnr( cv2Read(srPath), cv2Read('./test2.png'), data_range=255) ) 
# srPIL = imageRead( srPath )
# print(srPIL.shape)
# print( srPIL[0,0,0] )
# srtensor = transforms.ToTensor()( srPIL )

# srPIL2 = np.array( transforms.ToPILImage()( srtensor.clip(0,1) ).convert('RGB') )
# print( srPIL2.shape )
# print( srPIL2[0,0,0])


srCV2 = cv2Read( srPath )
# print( srCV2.shape )
# print( srCV2.dtype )

srtensor = transforms.ToTensor()( cv2.cvtColor(srCV2 , cv2.COLOR_BGR2RGB) )

srcv2_ = srtensor.numpy()
srcv2_ = np.transpose( srcv2_[ [2,1,0],:,:], (1,2,0) )
srcv2_ = ( srcv2_ * 255.0).round()
srcv2_ = srcv2_.astype( np.uint8 )
# cv2.imwrite("./test.png", srcv2_)
srcv2_ = srtensor.mul_(255.0).permute(1,2,0).numpy()
srcv2_ = cv2.cvtColor( srcv2_, cv2.COLOR_RGB2BGR )
# cv2.imwrite("./test2.png", srcv2_)

# print( srcv2_.shape)
# print( srcv2_[ [2,1,0],:,:].shape ) 

# srCV2_ = srtensor.mul_(255).permute(1,2,0).numpy()
# print(srCV2_[0,0,0])
# srCV2_ = cv2.cvtColor( srCV2_, cv2.COLOR_RGB2BGR )
# cv2.imwrite("./test.png", srCV2_)
#print( hrImage.dtype )
#print( calc_psnr(sr=srImage, hr=hrImage) )
#print( psnr(hrImage, srImage, data_range=255) )