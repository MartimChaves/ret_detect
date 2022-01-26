import os
os.chdir('C:/Users/Martim_Pc/Desktop/DACO_fin')

from Unet import Unet

import numpy as np
import cv2.cv2 as cv2
from skimage.measure import label, regionprops
from skimage.transform import downscale_local_mean,hough_ellipse
from skimage.filters.rank import entropy, mean
from skimage.filters import gabor, gabor_kernel
from skimage.morphology import disk, skeletonize, thin
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.draw import ellipse_perimeter
from skimage import data, color, img_as_ubyte
from matplotlib import pyplot as plt
import pickle
import math

import time

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt



def myShowImage(img,name = "from_show_function"):
    cv2.imshow(name, img) 

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    return

def removeBackground(img):
    maskRed = img[...,0]>30
    maskGreen = img[...,1]>30
    maskBlue = img[...,2]>30
    mask1 = np.logical_or(maskRed,maskGreen)
    maskFinal = np.logical_or(mask1,maskBlue)
    zeros = np.zeros(img.shape)
    zeros[maskFinal] = img[maskFinal]
    zeros = zeros.astype(np.uint8)
    img = np.copy(zeros)

    return img

testImages = ['20051020_44982_0100_PP.tif',
            '20051019_38557_0100_PP.tif',
            '20051213_62383_0100_PP.tif',
            'IDRiD_14.jpg',
            'OD0375EY.JPG']

def getVesselsUtils(imgPath):
    img = cv2.imread(imgPath)#,cv2.CV_8UC1)

    img = removeBackground(img)

    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # BGR - blue: 0; green: 1; red: 2
    resized = resized.astype(np.uint8) 

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    img_out1 = clahe.apply(resized[...,1])
    
    kernel_ones = np.ones([25,25],np.uint8)

    bh_1_cv = cv2.morphologyEx(img_out1, cv2.MORPH_BLACKHAT,kernel_ones)
    
    int_vs = clahe.apply(bh_1_cv)

    _,thresh2 = cv2.threshold(int_vs,60,255,cv2.THRESH_BINARY) # thresh2 is vessels segmentation used in OD segmentation

    labels, _ = label(thresh2, neighbors=8, background = 0, return_num = True)
    regions = regionprops(labels)

    for region in regions:
        value = labels[region['coords'][0][0],region['coords'][0][1]]
        #circularity = (4*math.pi*region['area']) / (region['perimeter']**2)
        bboxAreRel = region['area']/region['bbox_area'] 
        if region['area'] < 10 or (bboxAreRel > 0.35): #circularity > 0.3 and 
            removeIndexs = np.where(labels==value)
            labels[removeIndexs[0],removeIndexs[1]] = 0

    labels[labels > 0] = 1
    labelsImg = np.multiply(np.array(labels, np.uint8),255) # labelsImg = segmented relevant vessels

    myShowImage(labelsImg)

    # get skeleton of image
    doublT_small = downscale_local_mean(labels,(2,2))
    myShowImage(doublT_small)

    skeleton = skeletonize(doublT_small)

    skel = skeleton * 1
    skel = skel.astype(np.uint8)
    skel = np.multiply(skel,255)

    myShowImage(skel)


    threshAcc = 40
    for k in range(1,6):
        try:
            #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            #doublT_small = color.gray2rgb(img_as_ubyte(doublT_small))
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

            break

        except:
            threshAcc = threshAcc - 10
            if k == 5:
                threshAcc = threshAcc + 5

    ellipseMask = np.zeros(doublT_small.shape)
    ellipseMask[abs(cy-1),abs(cx-1)] = 1


    elipsResized = cv2.resize(ellipseMask, dsize=dim, interpolation=cv2.INTER_CUBIC)
    #elipsResized = np.average(elipsResized,axis = 2) # 3 channels -> 1 channel
    elipsResized[elipsResized>0.5] = 1
    elipsResized[elipsResized<1] = 0
    elipsResized = thin(elipsResized)
    elipsResized = elipsResized*1
    elipsImage = (elipsResized*255).astype(np.uint8)
    myShowImage(elipsImage)

    entr_img = entropy(labelsImg, disk(5))
    myShowImage(entr_img)

    vessels = np.copy(thresh2)
    ellipse = np.copy(elipsResized)
    entropyVessels = np.copy(entr_img)
    return vessels, entropyVessels, ellipse

def getDistanceArray(height,width):
    indices_Arr = np.indices((height,width)).transpose((1,2,0))
    centreCoords = np.array([height/2,width/2])
    distance_Arr = np.sqrt(np.add(np.power(indices_Arr[...,0]-centreCoords[0],2),np.power(indices_Arr[...,1]-centreCoords[1],2)))
    normDistance_Arr = distance_Arr / np.max(distance_Arr)
    normDistanceColumn_Arr = np.squeeze(normDistance_Arr.reshape([1,normDistance_Arr.shape[0]*normDistance_Arr.shape[1]])).T

    return normDistanceColumn_Arr

def reshapeFeature(img,featureSize,normalize=True):
    feature = np.squeeze(img.reshape([1,featureSize])).T
    if normalize:
        feature = (feature-np.average(feature)) / np.std(feature)
    return feature

def newBBcoords(img_pred_Log,test_image):
    # returns coordinates of the bounding box for the region with the largest area

    kernel_ones = np.ones([3,3],np.uint8)
    closing_Log = cv2.morphologyEx(img_pred_Log, cv2.MORPH_CLOSE, kernel_ones)

    labelsLog, numLog = label(closing_Log, neighbors=8, background = 0, return_num = True)
    regionsLog = regionprops(labelsLog)

    areasLog = [region['area'] for region in regionsLog]
    areasLogArr = np.array(areasLog)
    maxIndex = np.argmax(areasLogArr)

    value = labelsLog[regionsLog[maxIndex]['coords'][0][0],regionsLog[maxIndex]['coords'][0][1]]
    labelsLog[labelsLog != value] = 0
    labelsLog[labelsLog == value] = 1

    labelsImg = np.multiply(np.array(labelsLog, np.uint8),255)
    #myShowImage(labelsImg)

    sizeBoxX = regionsLog[maxIndex]['bbox'][3]-regionsLog[maxIndex]['bbox'][1]
    sizeBoxY = regionsLog[maxIndex]['bbox'][2]-regionsLog[maxIndex]['bbox'][0]

    coordsBbox = list(regionsLog[maxIndex]['bbox'])
    if sizeBoxX <= 0.4 * img_pred_Log.shape[1]:
        newSizeBoxX = 0.3 /  (sizeBoxX / img_pred_Log.shape[1])
        coordsBbox[1] = coordsBbox[1] - sizeBoxX*(0.5*(newSizeBoxX-1)) 
        coordsBbox[3] = coordsBbox[3] + sizeBoxX*(0.5*(newSizeBoxX-1)) 

    if sizeBoxY <= 0.4 * img_pred_Log.shape[0]:
        newSizeBoxY = 0.5 /  (sizeBoxY / img_pred_Log.shape[0])
        coordsBbox[0] = coordsBbox[0] - sizeBoxY*(0.5*(newSizeBoxY-1))
        coordsBbox[2] = coordsBbox[2] + sizeBoxY*(0.5*(newSizeBoxY-1))

    if coordsBbox[0] < 0:
        coordsBbox[0] = 0
    if coordsBbox[1] < 0:
        coordsBbox[1] = 0
    if coordsBbox[2] > test_image.shape[0]:
        coordsBbox[2] = test_image.shape[0] - 1
    if coordsBbox[3] > test_image.shape[1]:
        coordsBbox[3] = test_image.shape[1] - 1

    coordsBboxInt = [round(x) for x in coordsBbox]

    return coordsBboxInt

def getLargestAreaEcentroid(img_pred_Log):
    # returns mask with the regions with the largest area, coords of centroid and radius

    kernel_ones = np.ones([3,3],np.uint8)
    closing_Log = cv2.morphologyEx(img_pred_Log, cv2.MORPH_CLOSE, kernel_ones)

    labelsLog, numLog = label(closing_Log, neighbors=8, background = 0, return_num = True)
    regionsLog = regionprops(labelsLog)

    areasLog = [region['area'] for region in regionsLog]
    areasLogArr = np.array(areasLog)
    maxIndex = np.argmax(areasLogArr)

    value = labelsLog[regionsLog[maxIndex]['coords'][0][0],regionsLog[maxIndex]['coords'][0][1]]
    labelsLog[labelsLog != value] = 0
    labelsLog[labelsLog == value] = 1

    centreCoords = np.round(regionsLog[maxIndex]['centroid'])
    centreCoords = centreCoords.astype(np.uint)

    radius = (regionsLog[maxIndex]['major_axis_length'] + regionsLog[maxIndex]['minor_axis_length']) / 4
    colsCoord = [regionsLog[maxIndex]['bbox'][1],regionsLog[maxIndex]['bbox'][3]]

    labelsArr = np.array(labelsLog)

    return labelsArr, centreCoords, radius, colsCoord

def gaborFilter(img_in):
    filtered_ims = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (6, 7):
            for frequency in (0.06, 0.07):
                filt_real, _ = gabor(img_in, frequency, theta=theta,sigma_x=sigma, sigma_y=sigma) # _ = imaginary part
                filtered_ims.append(filt_real)

    filtered_ims_arr = np.array(filtered_ims)
    return filtered_ims_arr


def getFeature2(greenChannel):

    filtered_ims_arr = gaborFilter(greenChannel)

    mean = filtered_ims_arr[0]
    for k in range(1,16):
        mean = np.add(mean,filtered_ims_arr[k])

    mean = np.divide(mean,16)
    mean = mean.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    imgContrasted = clahe.apply(mean)

    maxVal = np.max(imgContrasted)
    feature2 = np.multiply(np.divide(imgContrasted,maxVal),255)
    feature2 = feature2.astype(np.uint8)


    return feature2

def getFeature3(elipsResized,Od_res,resized):

    Od_uint8 = np.copy(Od_res.astype(np.uint8))
    Od_uint8 = np.divide(Od_uint8,np.max(Od_uint8)).astype(np.uint8)
    testDistance = distance_transform_edt(1 - Od_uint8)
    testDistance = np.multiply(testDistance,255/np.max(testDistance))
    testDistance = testDistance.astype(np.uint8)
    distanceToOd = 255-testDistance
    distanceToOd[distanceToOd >220] = 255
    distanceToOd[distanceToOd <=220] = 0
    vesselsRelevantLine = np.logical_and(distanceToOd,elipsResized)
    vesselsRelevantLine = vesselsRelevantLine*1

    distanceToVesselLine = distance_transform_edt(1 - vesselsRelevantLine)
    distanceToVesselLine = np.multiply(distanceToVesselLine,255/np.max(distanceToVesselLine))
    distanceToVesselLine = distanceToVesselLine.astype(np.uint8)
    distanceToVesselLine = 255-distanceToVesselLine
    distanceToVesselLine[distanceToVesselLine >220] = 255
    distanceToVesselLine[distanceToVesselLine <=220] = 0
    distanceToVesselLine = np.logical_or(distanceToVesselLine,Od_uint8)

    greenChannel = np.copy(resized[...,1])
    vesselLine_indxs=np.where(distanceToVesselLine!=0)
    #greenChannel[vesselLine_indxs] = 0

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    greenContrasted = clahe.apply(greenChannel)
    
    greenContrasted = np.multiply(greenContrasted,255/np.max(greenContrasted))
    greenContrasted = greenContrasted.astype(np.uint8)

    greenContrasted[vesselLine_indxs] = 0

    return greenContrasted

def getFeature4(resized):

    radius = 1
    n_points = 8 * radius
    GC = np.copy(resized[...,1])
    lbp = local_binary_pattern(GC, n_points, radius,method="ror")
    step = int(-1 * np.max(lbp)/10)

    feature4 = np.copy(lbp)
    th_feat4 = int(np.max(lbp)+(step*7))
    feature4[feature4 < th_feat4] = 0
    feature4[feature4 >= th_feat4] = 255
    feature4 = distance_transform_edt(255-feature4)
    feature4 = np.multiply(feature4,255/np.max(feature4))
    feature4 = feature4.astype(np.uint8)

    return feature4

def getFeatures567(img,height,width,scale_percent):

    scale_percent = int(100/scale_percent)

    feature5 = np.zeros([height,width])
    feature6 = np.zeros([height,width])
    feature7 = np.zeros([height,width])

    for m in range(width):
        for t in range(height):
            patch = np.copy(img[t*scale_percent:(t+1)*scale_percent,m*scale_percent:(m+1)*scale_percent,1])
            glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
            feature5[t,m]=greycoprops(glcm, 'dissimilarity')[0, 0] # good
            feature6[t,m]=greycoprops(glcm, 'contrast')[0, 0] # th-55 good
            feature7[t,m]=greycoprops(glcm, 'homogeneity')[0, 0] # ok

    feature5 = np.divide(feature5,np.max(feature5))
    feature5 = np.multiply(feature5,255)
    feature5 = feature5.astype(np.uint8)

    feature6[feature6<55]=0
    feature6[feature6>=55]=255
    feature6 = feature6.astype(np.uint8)

    feature7 = np.divide(feature7,np.max(feature7))
    feature7 = np.multiply(feature7,255)
    feature7 = 255-feature7
    feature7 = feature7.astype(np.uint8)

    return feature5, feature6, feature7

def calculateHue(img):

    RC = img[...,2]/255 # 
    GC = img[...,1]/255
    BC = img[...,0]/255
    
    num = np.multiply(np.add(np.subtract(RC,GC),np.subtract(RC,BC)),0.5)
    denom = np.power(np.add(np.power(np.subtract(RC,GC),2),np.multiply(np.subtract(RC,BC),np.subtract(GC,BC))),0.5)

    angle = np.divide(num,np.add(denom,0.000000000001))
    H = np.multiply(np.arccos(angle),57.295779513)

    H[BC>GC] = 360 - H[BC>GC]

    H = np.divide(H,360)
    H[H>0.3]=0

    return H

def getFeature8(resized):

    Hue = calculateHue(np.copy(resized))
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    Hue = np.divide(Hue,np.max(Hue))
    Hue = Hue * 255
    Hue = Hue.astype(np.uint8)
    contrastedHue = clahe.apply(Hue)

    return contrastedHue

def getBoundingBlackBars(img):

    firstCol = np.min(np.where(img!=0)[1])
    lastCol = np.max(np.where(img!=0)[1])

    finalWidth = int(lastCol-firstCol)
    offsetRows = int(round((finalWidth-img.shape[0])/2))

    return finalWidth, offsetRows, firstCol, lastCol

def removeBlackBarsRetina(img,finalWidth,offsetRows,firstCol,lastCol,RGB=True):

    #Get a squared image
    #Retina image 
    
    if RGB:
        squareToBePatched = np.zeros([img.shape[0],finalWidth,3])
        squareToBePatched[::,::] = np.copy(img[::,firstCol:lastCol])
    else:
        squareToBePatched = np.zeros([img.shape[0],finalWidth])
        squareToBePatched[::,::] = np.copy(img[::,firstCol:lastCol])

    return squareToBePatched

def reshapeFeatureForKnn(feature,imgHeight,imgWidth,normalize=True):
    feature = np.squeeze(feature.reshape([1,imgHeight*imgWidth])).T
    if normalize:
        feature = (feature-np.average(feature)) / np.std(feature)
    return feature


imgPath = testImages[4]
vessels, entropyVessels, ellipse = getVesselsUtils(imgPath)

# get OD region classifier
f = open('detectOdRegion/Classifiers/BayClf_OD_ROI_.pickle', 'rb')
bay_OD_Region = pickle.load(f)
f.close()

g = open('detectOdRegion/Classifiers/LogClf_OD_ROI_.pickle', 'rb')
log_OD_Region = pickle.load(g)
g.close()

print("Loaded region classifiers.")

# Get OD region
# ^get features - predict
img = cv2.imread(imgPath)#,cv2.CV_8UC1)
img = removeBackground(img)

intensityImg = cv2.imread(imgPath,cv2.CV_8UC1)

scale_percent = 25 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
featureSize = height * width
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
intensityResized = cv2.resize(intensityImg, dim, interpolation = cv2.INTER_AREA)

intensity = reshapeFeature(intensityResized,featureSize)
redChannel = reshapeFeature(np.copy(resized[...,2]),featureSize)
entropyImg = reshapeFeature(entropyVessels,featureSize)
normDistance = getDistanceArray(height,width)

X_val = np.column_stack((redChannel,entropyImg,normDistance,intensity))

# predicts
predictsBayes = bay_OD_Region.predict(X_val)
predictsLog = log_OD_Region.predict(X_val)

predictsBayes = predictsBayes.reshape([intensityResized.shape[0],intensityResized.shape[1]])#372,560]) # hardcoded - not good...
predictsLog = predictsLog.reshape([intensityResized.shape[0],intensityResized.shape[1]])#372,560])

# Giving precedence to the Log classifier
try:
    # from log classifier
    # get bbox coords of region
    coords = newBBcoords(predictsLog,intensityResized)
    # get regions
    redChan = np.copy(resized[...,2])
    segRC = np.array(redChan[coords[0]:coords[2],coords[1]:coords[3]])
    imgSegment = np.array(intensityResized[coords[0]:coords[2],coords[1]:coords[3]])
    vesslesSeg = np.array(vessels[coords[0]:coords[2],coords[1]:coords[3]])
    # smooth vessels of region
    kernel_ones = np.ones([3,3],np.uint8)
    vesslesSeg = cv2.morphologyEx(vesslesSeg, cv2.MORPH_DILATE, kernel_ones)
    indxsVesl = np.where(vesslesSeg != 0)  

    smoothVessels = np.copy(imgSegment)
    maxFiltered = maximum_filter(imgSegment, size=6)
    smoothVessels[indxsVesl[0],indxsVesl[1]] = np.multiply(maxFiltered[indxsVesl[0],indxsVesl[1]],0.97)
except:
    # from bayes classifier
    # do the same
    coords = newBBcoords(predictsBayes,intensityResized)
    # get regions
    redChan = np.copy(resized[...,2])
    segRC = np.array(redChan[coords[0]:coords[2],coords[1]:coords[3]])
    imgSegment = np.array(intensityResized[coords[0]:coords[2],coords[1]:coords[3]])
    vesslesSeg = np.array(vessels[coords[0]:coords[2],coords[1]:coords[3]])
    # smooth vessels of region
    kernel_ones = np.ones([3,3],np.uint8)
    vesslesSeg = cv2.morphologyEx(vesslesSeg, cv2.MORPH_DILATE, kernel_ones)
    indxsVesl = np.where(vesslesSeg != 0)  

    smoothVessels = np.copy(imgSegment)
    maxFiltered = maximum_filter(imgSegment, size=15)
    smoothVessels[indxsVesl[0],indxsVesl[1]] = np.multiply(maxFiltered[indxsVesl[0],indxsVesl[1]],0.97)

# segment OD
bay = open('segmentOD/Classifiers/BayClf_OD_SEG_.pickle', 'rb')
clf_bayes_ODseg = pickle.load(bay)
bay.close()

featSize = smoothVessels.shape[0]*smoothVessels.shape[1]

intensityColumn_Arr = np.squeeze(smoothVessels.reshape([1,featSize])).T
intensityColumn_Arr = (intensityColumn_Arr-np.average(intensityColumn_Arr)) / np.std(intensityColumn_Arr)
segRC = np.squeeze(segRC.reshape([1,featSize])).T
segRC = segRC/255

X_val = np.column_stack((intensityColumn_Arr,segRC))

predictsBayes = clf_bayes_ODseg.predict(X_val)

# Get Fovea ROI
maskPred = predictsBayes.reshape([smoothVessels.shape[0],smoothVessels.shape[1]])

useUnet = True
if useUnet:
    model = Unet(2,256)
    model.load_weights("model.h5")
    modelInput = np.copy(resized[coords[0]:coords[2],coords[1]:coords[3],::])
    initialShape = modelInput.shape
    modelInput = cv2.resize(modelInput, (256,256), interpolation = cv2.INTER_AREA)
    inputReal = []
    modelInput = modelInput/255
    R = modelInput[...,2]
    G = modelInput[...,1]
    B = modelInput[...,0]
    modelInput = np.zeros([256,256,3])
    modelInput[...,0] = R
    modelInput[...,1] = G
    modelInput[...,2] = B
    inputReal.append(modelInput)
    inputReal = np.array(inputReal)
    pred_mask = model.predict([inputReal])[0]*255
    pred_mask[pred_mask > 0.5] = 1
    pred_mask[pred_mask <= 0.5] = 0
    predUnet = pred_mask[...,1]
    predUnet = cv2.resize(predUnet, (initialShape[1],initialShape[0]), interpolation = cv2.INTER_AREA)
    #maskPred = predUnet
    myShowImage(predUnet)

finalMask, centroidCoords, radius, colsCoord = getLargestAreaEcentroid(maskPred)

originalDims = np.zeros(intensityResized.shape)

finalMaskImg = np.multiply(finalMask,255)
finalMaskImg[centroidCoords[0],centroidCoords[1]] = 255
originalDims[coords[0]:coords[2],coords[1]:coords[3]] = finalMaskImg # OD MASK HERE

coordsReal =[centroidCoords[0] + coords[0],centroidCoords[1] + coords[1]] 
colsCoordReal = [colsCoord[0] + coords[1],colsCoord[1] + coords[1]]

elipsIndexs = np.where(ellipse != 0)

if abs(coordsReal[1]-np.min(elipsIndexs[1])) < abs(coordsReal[1]-np.max(elipsIndexs[1])):
    #isleft -> walk right
    #relevantColumn = coordsReal[1] + 30 # based on centroid
    relevantColumn = colsCoordReal[1] - 10 # based on 
    columnROI_f = [coordsReal[1] + round(3*radius),coordsReal[1] + round(6*radius)]

else:
    #isright -> walk left
    #relevantColumn = coordsReal[1] - 30
    relevantColumn = colsCoordReal[0] + 10
    columnROI_f = [coordsReal[1] - round(6*radius),coordsReal[1] - round(3*radius)]

relevantRows = np.where(ellipse[...,relevantColumn]!=0)

foveaBBox = (relevantRows[0][0],columnROI_f[0],relevantRows[0][-1],columnROI_f[1])

checkResults = np.copy(intensityResized)
checkResults[relevantRows[0][0]:relevantRows[0][-1],columnROI_f[0]] = 0 # 1 - columnROI_f[0]
checkResults[relevantRows[0][0]:relevantRows[0][-1],columnROI_f[1]] = 0 # 3 - columnROI_f[1]
checkResults[relevantRows[0][0],columnROI_f[0]:columnROI_f[1]] = 0 # 0 - relevantRows[0][0]
checkResults[relevantRows[0][-1],columnROI_f[0]:columnROI_f[1]] = 0 # 2 - relevantRows[0][-1]

myShowImage(checkResults)

# Get fovea coordinates
fovea_region = np.copy(intensityResized[foveaBBox[0]:foveaBBox[2],foveaBBox[1]:foveaBBox[3]])
bboxShape = fovea_region.shape

greenChan = np.copy(resized[...,1])
foveaRegionGC = greenChan[foveaBBox[0]:foveaBBox[2],foveaBBox[1]:foveaBBox[3]]

clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
clahePrep = np.multiply(np.copy(foveaRegionGC),255)
clahePrep = clahePrep.astype(np.uint8)
highContrast = clahe.apply(clahePrep)

fovea_region = np.squeeze(fovea_region.reshape([1,bboxShape[0]*bboxShape[1]])).T
fovea_region = (fovea_region-np.average(fovea_region)) / np.std(fovea_region)

foveaRegionGC = np.squeeze(foveaRegionGC.reshape([1,bboxShape[0]*bboxShape[1]])).T
foveaRegionGC = (foveaRegionGC-np.average(foveaRegionGC)) / np.std(foveaRegionGC)

highContrast = np.squeeze(highContrast.reshape([1,bboxShape[0]*bboxShape[1]])).T
highContrast = (highContrast-np.average(highContrast)) / np.std(highContrast)

X_val = np.column_stack((fovea_region,foveaRegionGC,highContrast))

bay = open('segmentFoveaRegion/Classifiers/BayClf_Fovea_.pickle', 'rb')
clf_bayes_Fovea = pickle.load(bay)
bay.close()

predictsBayes = clf_bayes_Fovea.predict(X_val)
img_pred_Bayes = predictsBayes.reshape(bboxShape)

# If there is no response, choose fovea centre the centre of region of interest
try:
    finalMask, centroidCoords, radiusFovea, colsCoord = getLargestAreaEcentroid(img_pred_Bayes)
    if centroidCoords.size == 0:
        finalMask = np.zeros(img_pred_Bayes.shape)
        heightROI = img_pred_Bayes.shape[0]
        widthROI  = img_pred_Bayes.shape[1]
        finalMask[round(heightROI/2),round(widthROI/2)] = 1
        centroidCoords = np.array([round(heightROI/2),round(widthROI/2)])
except:
    finalMask = np.zeros(img_pred_Bayes.shape)
    heightROI = img_pred_Bayes.shape[0]
    widthROI  = img_pred_Bayes.shape[1]
    finalMask[round(heightROI/2),round(widthROI/2)] = 1
    centroidCoords = np.array([round(heightROI/2),round(widthROI/2)])

realCentroidCoords = [centroidCoords[0] + foveaBBox[0],centroidCoords[1] + foveaBBox[1]]
print("CentroidCoords: ",realCentroidCoords)
checkResults[realCentroidCoords[0],realCentroidCoords[1]]=255
myShowImage(checkResults)

# we know the OD radius and fovea coords

# get the exudates

# get features
feature1 = np.copy(resized[...,1])
Od_indxs = np.where(originalDims!=0)
feature1[Od_indxs] = 0
myShowImage(feature1)

greenChannel = np.copy(resized[...,1])
feature2 = getFeature2(greenChannel) 
myShowImage(feature2)

feature3 = getFeature3(ellipse,originalDims,resized)
myShowImage(feature3)

feature4 = getFeature4(resized)
myShowImage(feature4)

scale_percentGLCM = 6 # percent of original size
widthGLCM = int(img.shape[1] * scale_percentGLCM / 100)
heightGLCM = int(img.shape[0] * scale_percentGLCM / 100)

feature5, feature6, feature7 = getFeatures567(img,heightGLCM,widthGLCM,scale_percentGLCM)

feature5 = cv2.resize(feature5, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
feature6 = cv2.resize(feature6, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
feature7 = cv2.resize(feature7, dsize=(width,height), interpolation=cv2.INTER_CUBIC)

myShowImage(feature5)
myShowImage(feature6)
myShowImage(feature7)

feature8 = getFeature8(resized)
myShowImage(feature8)

lda = open('LDA_featureExtract.pickle', 'rb')
Lda_featureExtract = pickle.load(lda)
lda.close()

finalWidth, offsetRows, firstCol, lastCol = getBoundingBlackBars(resized)
feature1 = removeBlackBarsRetina(feature1,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature2 = removeBlackBarsRetina(feature2,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature3 = removeBlackBarsRetina(feature3,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature4 = removeBlackBarsRetina(feature4,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature5 = removeBlackBarsRetina(feature5,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature6 = removeBlackBarsRetina(feature6,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature7 = removeBlackBarsRetina(feature7,finalWidth,offsetRows,firstCol,lastCol,RGB=False)
feature8 = removeBlackBarsRetina(feature8,finalWidth,offsetRows,firstCol,lastCol,RGB=False)

imgHeight = feature1.shape[0]
imgWidth = feature1.shape[1]
featureSize = imgHeight * imgWidth

feature1 = reshapeFeatureForKnn(feature1,imgHeight,imgWidth)
feature2 = reshapeFeatureForKnn(feature2,imgHeight,imgWidth)
feature3 = reshapeFeatureForKnn(feature3,imgHeight,imgWidth)
feature4 = reshapeFeatureForKnn(feature4,imgHeight,imgWidth)
feature5 = reshapeFeatureForKnn(feature5,imgHeight,imgWidth)
feature6 = reshapeFeatureForKnn(feature6,imgHeight,imgWidth)
feature7 = reshapeFeatureForKnn(feature7,imgHeight,imgWidth)
feature8 = reshapeFeatureForKnn(feature8,imgHeight,imgWidth)

X_val_featExtract = np.column_stack((feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8)) 

feature9 = Lda_featureExtract.predict(X_val_featExtract)

X_val_Final= np.column_stack((feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9)) 

kNN_classifier = open('kNN_exudates_.pickle', 'rb')
clfKnn = pickle.load(kNN_classifier)
kNN_classifier.close()

Z = clfKnn.predict(X_val_Final)
prediction = Z.reshape([imgHeight,imgWidth])
myShowImage(prediction)


indxsTrue = np.where(prediction!=0)

if indxsTrue[0].size == 0:
    print("DME0")
else:
    print("DME1 or DME2")
