import cv2.cv2 as cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt

import skimage.io as io
from skimage.transform import downscale_local_mean
from skimage import data, color, img_as_ubyte
from skimage.filters.rank import entropy, mean
from skimage.morphology import disk, skeletonize, thin
from skimage.feature import canny
from skimage.transform import hough_ellipse, hough_circle, hough_circle_peaks
from skimage.draw import ellipse_perimeter, circle_perimeter
from skimage.measure import label, regionprops
from skimage.filters import median, threshold_otsu

import time
import math

from PIL import ImageFilter
from scipy.ndimage.filters import maximum_filter

def myShowImage(img,name = "from_show_function"):
    cv2.imshow(name, img) 

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    return

def calculateHue(RC_orig,GC_orig,BC_orig):

    RC = np.divide(np.copy(RC_orig),255)
    GC = np.divide(np.copy(GC_orig),255)
    BC = np.divide(np.copy(BC_orig),255)

    RC = np.subtract(np.multiply(RC,1.11),0.11)
    GC = np.subtract(np.multiply(GC,1.11),0.11)
    BC = np.subtract(np.multiply(BC,1.11),0.11)

    RC[RC < 0] = 0
    GC[GC < 0] = 0
    BC[BC < 0] = 0

    num = np.multiply(np.add(np.subtract(RC,GC),np.subtract(RC,BC)),0.5)
    denom = np.power(np.add(np.power(np.subtract(RC,GC),2),np.multiply(np.subtract(RC,BC),np.subtract(GC,BC))),0.5)

    H = np.multiply(np.arccos(np.divide(num,np.add(denom,0.0000000001))),57.295779513)

    H[BC>GC] = 360 - H[BC>GC]

    H = np.divide(H,360)

    return H

def preProcessing(Hue,intensityValues = [0.2, 1, 7]):

    H_neg = np.zeros(Hue.shape)

    H_neg = np.subtract(1,Hue)

    H_neg = np.power(np.subtract(np.divide(H_neg,1.25),0.25),30)

    return H_neg

def prepRocs(img_in,clip_limit = 0.01):

    img_out = np.zeros(img_in.shape)
    #intensityValues = [0.2, 1, 7]

    img_in = np.multiply(np.divide(img_in,np.average(img_in)),255)

    img_in = img_in.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    img_out = clahe.apply(img_in)

    # img_out = cv2.equalizeHist(img_in)

    # img_out = np.power(np.divide(np.subtract(img_in,intensityValues[0]),intensityValues[1]-intensityValues[0]),intensityValues[2])

    return img_out

if __name__ == "__main__":

    image_arr = []
    mask_arr = []
    entr_arr = []
    elips_arr = []
    vessels = []
    showImageCV = False
    showImageCompare = False
    showMask = False


    for i in range(1,41):

        imgPath = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '.jpg' 
        imgPathMasks = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '_OD.tif' 

        img = cv2.imread(imgPath)#,cv2.CV_8UC1)

        scale_percent = 25 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # BGR - blue: 0; green: 1; red: 2
        resized = np.subtract(np.multiply(resized,(255/230)),28)
        resized[resized < 0] = 0
        resized = resized.astype(np.uint8)

        #resized = 

        '''Hue = calculateHue(resized[...,2],resized[...,1],resized[...,0])
        proc1 = preProcessing(Hue)
        proc2 = prepRocs(proc1)

        myShowImage(Hue,"Hue")
        myShowImage(np.multiply(proc1,255),"proc1")
        myShowImage(proc2,"proc2")

        '''
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
        img_out1 = clahe.apply(resized[...,1])

        kernel_ones = np.ones([25,25],np.uint8)
        kernel_zeros = np.zeros([25,25],np.uint8)

        closing_ones = cv2.morphologyEx(img_out1, cv2.MORPH_CLOSE, kernel_ones)
        closing_zeros = cv2.morphologyEx(img_out1, cv2.MORPH_ERODE, kernel_zeros)
        bh_1_cv = cv2.morphologyEx(img_out1, cv2.MORPH_BLACKHAT,kernel_ones)
        bh_cv = cv2.morphologyEx(img_out1, cv2.MORPH_BLACKHAT,kernel_zeros)

        # minImg = np.zeros(img_out1.shape,np.uint8)

        '''for i in range(2,img_out1.shape[0]-1):
            for j in range(2,img_out1.shape[1]-1):
                minImg[i,j] = np.min(img_out1[i-2:i+2,j-2:j+2])'''



        bottomHatOnes = closing_ones - img_out1 


        img_out2 = cv2.equalizeHist(resized[...,1])

      
        int_vs = clahe.apply(bh_1_cv)
        #myShowImage(int_vs,"VS_int")

        # myShowImage(minImg,"Min")
        # test = np.subtract(minImg,img_out1)
        # myShowImage(test,"Min-clahe")

        # ret,thresh1 = cv2.threshold(minImg,60,255,cv2.THRESH_BINARY_INV)
        ret2,thresh2 = cv2.threshold(int_vs,60,255,cv2.THRESH_BINARY)

        #myShowImage(thresh2) 

        kernel_ones_small = np.ones([3,3],np.uint8)
        closedThresh = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel_ones_small)

        labels, num = label(thresh2, neighbors=8, background = 0, return_num = True)
        regions = regionprops(labels)

        for region in regions:
            value = labels[region['coords'][0][0],region['coords'][0][1]]
            circularity = (4*math.pi*region['area']) / (region['perimeter']**2)
            bboxAreRel = region['area']/region['bbox_area'] 
            if region['area'] < 10 or (bboxAreRel > 0.35): #circularity > 0.3 and 
                removeIndexs = np.where(labels==value)
                labels[removeIndexs[0],removeIndexs[1]] = 0

        labels[labels > 0] = 1
        labelsImg = np.multiply(np.array(labels, np.uint8),255)

        #myShowImage(labelsImg)

        getSkeleton = False
        if getSkeleton:

            doubleThresh2 = np.divide(thresh2,255)

            doublT_small = downscale_local_mean(labels,(4,4))

            skeleton = skeletonize(doublT_small)

            skel = skeleton * 1
            skel = skel.astype(np.uint8)
            skel = np.multiply(skel,255)
            #myShowImage(skel)

            #edges = canny(thresh2, sigma=2.0,low_threshold=0.55, high_threshold=0.8)

        #try:
        getCircles = False
        if getCircles:
            hough_radii = np.arange(50, skeleton.shape[1], 2)
            hough_res = hough_circle(skeleton, hough_radii)
            
            # Select the most prominent 3 circles
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=3)

            # Draw them
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            doublT_small = color.gray2rgb(img_as_ubyte(doublT_small))

        '''
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=doublT_small.shape)
            doublT_small[circy, circx] = (220, 20, 20)

        
        ax.imshow(doublT_small, cmap=plt.cm.gray)
        plt.show()
        plt.clf()
        '''
        getEllipsis = False
        if getEllipsis:
            threshAcc = 40
            for k in range(1,6):
                try:
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
                    doublT_small = color.gray2rgb(img_as_ubyte(doublT_small))
                    tic = time.time()
                    result = hough_ellipse(skeleton, accuracy=20, threshold=threshAcc, min_size=50, max_size=None) # thresh = 30
                    result.sort(order='accumulator')
                    aSizeResult_Arr = np.array(result['a'])
                    bSizeResult_Arr = np.array(result['b'])
                    aIndex = np.where(aSizeResult_Arr > 0)
                    bIndex = np.where(bSizeResult_Arr > 0)
                    
                    relevantIndexs = np.intersect1d(aIndex,bIndex) 

                    axisRelation = np.divide(aSizeResult_Arr[relevantIndexs],bSizeResult_Arr[relevantIndexs])
                    goodRelationIndexs = np.where(axisRelation<1.5)

                    ellipsLargest = np.max(relevantIndexs[goodRelationIndexs])
                    toc = time.time()
                    ellapsedTime = toc-tic
                    print(ellapsedTime)

                    best = list(result[ellipsLargest])
                    yc, xc, a, b = [int(round(x)) for x in best[1:5]]

                    orientation = best[5]
                    #print(best[0])
                    # Draw the ellipse on the original image
                    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
                    # Draw the edge (white) and the resulting ellipse (red)
                    outsideX = np.where(cx>doublT_small.shape[1])
                    preX = np.where(cx<0)
                    outsideY = np.where(cy>doublT_small.shape[0])
                    preY = np.where(cy<0)
                    cx[outsideX] = doublT_small.shape[1]
                    cy[outsideY] = doublT_small.shape[0]
                    cx[preX] = 0
                    cy[preY] = 0

                    # Estimated parameters for the ellipse
                    '''lst=range(0,len(result))
                    for i in lst:
                        best = list(result[i])
                        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
                        if a == 0 or b == 0:# or not (xc > cx[0]-20 and xc < cx[0]+20 and yc > cy[0]-20 and yc < cy[0]+20):
                            continue
                        else:
                            orientation = best[5]
                            #print(best[0])
                            # Draw the ellipse on the original image
                            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
                            # Draw the edge (white) and the resulting ellipse (red)
                            outsideX = np.where(cx>doublT_small.shape[1])
                            preX = np.where(cx<0)
                            outsideY = np.where(cy>doublT_small.shape[0])
                            preY = np.where(cy<0)
                            cx[outsideX] = doublT_small.shape[1]
                            cy[outsideY] = doublT_small.shape[0]
                            cx[preX] = 0
                            cy[preY] = 0
                            #break'''

                    doublT_small[abs(cy-1), abs(cx-1)] = (220, 20, 20)
                    #ax.imshow(doublT_small, cmap=plt.cm.gray)
                    #plt.show()
                    break
                except:
                    threshAcc = threshAcc - 10
                    if k == 5:
                        threshAcc = threshAcc + 5

        #entr_img = entropy(labelsImg, disk(10))
        #ellipseMask = np.zeros(doublT_small.shape)
        #ellipseMask[abs(cy-1),abs(cx-1)] = 1

        readImageMask = False
        if readImageMask:
            #except:
            #   print("No good")
            

            # myShowImage(thresh1,"zeros_thresh")
            myShowImage(thresh2,"ones_thresh")

            kernel_ones_small = np.ones([3,3],np.uint8)
            # closed_min = cv2.morphologyEx(minImg, cv2.MORPH_CLOSE, kernel_ones_small)
            # eroded_min = cv2.morphologyEx(minImg, cv2.MORPH_ERODE, kernel_ones_small)

            eroded_thresh = cv2.morphologyEx(thresh2, cv2.MORPH_ERODE, kernel_ones_small)
            #myShowImage(eroded_thresh,"eroded_thresh")

            entr_img = entropy(eroded_thresh, disk(10))

            #myShowImage(entr_img)


            # myShowImage(closed_min,"closed_min")
            # myShowImage(eroded_min,"closed_min")


            
            print('Resized Dimensions : ',resized.shape)
            
            if showImageCV:
                for j in range(3):
                    cv2.imshow("Resized_image_" + str(i) + "_channel_" + str(j), resized[...,j]) 

                    cv2.waitKey(0) # waits until a key is pressed
                    cv2.destroyAllWindows() # destroys the window showing image

            img_skio = io.imread(imgPath) #, as_gray = True)
            img_skio = img_skio / 255
            img_resized = downscale_local_mean(img_skio,(4,4,1)) # RGB - red: 0; green: 1; blue: 2 # If as_gray=true, then resize = (4,4)

            img_skio_mask = io.imread(imgPathMasks, as_gray = True)
            img_skio_mask = img_skio_mask #/ 255
            img_resized_mask = downscale_local_mean(img_skio_mask,(4,4)) # RGB - red: 0; green: 1; blue: 2

            img_resized_mask[img_resized_mask == np.max(img_resized_mask)] = 1
            img_resized_mask[img_resized_mask < np.max(img_resized_mask)] = 0

            if showImageCompare:
                for j in range(3):
                    cv2.imshow("Resized_image_" + str(i) + "_channel_" + str(j), img_resized[...,j]) 

                    cv2.waitKey(0) # waits until a key is pressed
                    cv2.destroyAllWindows() # destroys the window showing image

                    '''cv2.imshow("Resized_image_" + str(i) + "_channel_" + str(j), resized[...,2-j]) 

                    cv2.waitKey(0) # waits until a key is pressed
                    cv2.destroyAllWindows() # destroys the window showing image'''

            if showMask:
                cv2.imshow("OD_Mask_" + str(i), img_resized_mask) 

                cv2.waitKey(0) # waits until a key is pressed
                cv2.destroyAllWindows() # destroys the window showing image

                cv2.imshow("Retina_" + str(i), img_resized) 

                cv2.waitKey(0) # waits until a key is pressed
                cv2.destroyAllWindows() # destroys the window showing image

            image_arr.append(img_resized[...,2])
            mask_arr.append(img_resized_mask)

        #entr_arr.append(entr_img)
        #elips_arr.append(ellipseMask)
        vessels.append(thresh2)

    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    entr_arr = np.array(entr_arr)
    elips_arr = np.array(elips_arr)
    vessels_arr = np.array(vessels)

    #np.save('image_arr_blue_channels.npy',image_arr)
    #np.save('mask_arr.npy',mask_arr)
    #np.save('entropy_arr.npy',entr_arr)
    #np.save('elips_arr.npy',elips_arr)
    #np.save('vessels_arr.npy',vessels_arr)

# To load images:
'''image_arr = np.load('image_arr.npy')
mask_arr = np.load('mask_arr.npy')'''