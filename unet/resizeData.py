import cv2.cv2 as cv2
import numpy as np
import os

def myShowImage(img,name = "from_show_function"):
    cv2.imshow(name, img) 

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    return

 
for i in range(1,41):
    os.chdir("C:/Users/Martim_Pc/Desktop/DACO/PROJECT_DACO/convNet/Unet/")
    imgPath = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '.jpg' 
    imgPathMasks = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '_OD.tif' 

    img = cv2.imread(imgPathMasks,cv2.CV_8UC1)

    scale_percent = 5.95 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # BGR - blue: 0; green: 1; red: 2
    resized = np.subtract(np.multiply(resized,(255/230)),28)
    resized[resized < 0] = 0
    resized = resized.astype(np.uint8)

    finalResized = np.array(resized)
    resFin = np.zeros([256,256])
    resFin[43:212,0:255]=finalResized
    resFin = np.multiply(resFin,255/56) # if masks
    resFin = resFin.astype(np.uint8)

    os.chdir("C:/Users/Martim_Pc/Desktop/DACO/PROJECT_DACO/convNet/Unet/masks/train")
    cv2.imwrite(str(i)+".png", resFin)

    #myShowImage(resFin)