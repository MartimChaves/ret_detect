import cv2.cv2 as cv2
import skimage.io as io
from skimage.transform import downscale_local_mean
import numpy as np
from model import *

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from images_to_arr import *

import pickle
import csv

def removeBackground(img_in):
    Img_backless = np.copy(img_in)
    Img_backless = np.subtract(np.multiply(Img_backless,1.11),0.11)
    Img_backless[Img_backless < 0] = 0
    return Img_backless

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
    if sizeBoxX <= 0.5 * img_pred_Log.shape[1]:
        newSizeBoxX = 0.3 /  (sizeBoxX / img_pred_Log.shape[1])
        coordsBbox[1] = coordsBbox[1] - sizeBoxX*(0.5*(newSizeBoxX-1)) 
        coordsBbox[3] = coordsBbox[3] + sizeBoxX*(0.5*(newSizeBoxX-1)) 

    if sizeBoxY <= 0.5 * img_pred_Log.shape[0]:
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



image_arr = np.load('image_arr.npy')
mask_arr = np.load('mask_arr.npy')
image_arr_red_channels = np.load('image_arr_red_channels.npy')
image_arr_green_channels = np.load('image_arr_green_channels.npy')
image_arr_blue_channels = np.load('image_arr_blue_channels.npy')
entropy = np.load('entropy_arr.npy')
elips = np.load('elips_arr.npy')
vessels = np.load('vessels_arr.npy')


test_image = np.zeros(image_arr[0].shape)
test_image_mask = np.zeros(mask_arr[0].shape)
test_img_RC = np.zeros(image_arr[0].shape)
test_img_GC = np.zeros(image_arr[0].shape)
test_img_BC = np.zeros(image_arr[0].shape)
entropy_arr = np.zeros(image_arr[0].shape)
elips_arr = np.zeros(image_arr[0].shape)

ODROILog = []
ODROIBay = []
getClassifiers = False
if getClassifiers:
    X_train = np.zeros([image_arr[0].shape[0]*image_arr[0].shape[1]*40,4])
    Y_train = np.zeros([image_arr[0].shape[0]*image_arr[0].shape[1]*40,1])
    for j in range(0,40):

        for i in range(0,40): # Get train data

            if i == j:
                continue

            test_image = image_arr[i]
            test_image_mask = mask_arr[i]

            labels, num = label(test_image_mask, neighbors=8, background = 0, return_num = True)
            regions = regionprops(labels)
            centreCoords = np.round(regions[0]['centroid'])
            centreCoords = centreCoords.astype(np.uint)

            centreMask = np.zeros(test_image_mask.shape)
            centreMask[centreCoords[0],centreCoords[1]] = 1

            #Change here!
            #test_image_mask = centreMask

            test_image_RC = image_arr_red_channels[i]
            test_image_GC = image_arr_green_channels[i]
            test_image_BC = image_arr_blue_channels[i]
            entropy_arr = entropy[i]
            elips_arr = elips[i]

            #test_image_RC = removeBackground(test_image_RC)
            #test_image = removeBackground(test_image)

            imageIndxs = np.where(test_image != 0)

            intensityColumn_Arr = np.squeeze(test_image.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
            intensityColumn_Arr = (intensityColumn_Arr-np.average(intensityColumn_Arr)) / np.std(intensityColumn_Arr)
            redChannel_Arr = np.squeeze(test_image_RC.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
            redChannel_Arr = (redChannel_Arr-np.average(redChannel_Arr)) / np.std(redChannel_Arr)
            entropy_arr = np.squeeze(entropy_arr.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
            #entropy_arr = (entropy_arr-np.average(entropy_arr)) / np.std(entropy_arr)

            # Distance Array
            indices_Arr = np.indices((test_image.shape[0],test_image.shape[1])).transpose((1,2,0))
            centreCoords = np.array([test_image.shape[0]/2,test_image.shape[1]/2])
            distance_Arr = np.sqrt(np.add(np.power(indices_Arr[...,0]-centreCoords[0],2),np.power(indices_Arr[...,1]-centreCoords[1],2)))
            normDistance_Arr = distance_Arr / np.max(distance_Arr)
            normDistanceColumn_Arr = np.squeeze(normDistance_Arr.reshape([1,normDistance_Arr.shape[0]*normDistance_Arr.shape[1]])).T


            X_train[i*image_arr[0].shape[0]*image_arr[0].shape[1]:(i+1)*image_arr[0].shape[0]*image_arr[0].shape[1],...] = np.column_stack((redChannel_Arr,entropy_arr,normDistanceColumn_Arr, intensityColumn_Arr))#,
            Y_train[i*image_arr[0].shape[0]*image_arr[0].shape[1]:(i+1)*image_arr[0].shape[0]*image_arr[0].shape[1],0] = np.squeeze(test_image_mask.reshape([1,test_image_mask.shape[0]*test_image_mask.shape[1]])).T


        X_train_2 = X_train
        y_train_2 = Y_train

        clf_bayes = GaussianNB()
        clf_bayes.fit(X_train_2,y_train_2)

        paramsBayes = clf_bayes.get_params

        # Logistic regression
        clf_log = LogisticRegression()
        clf_log.fit(X_train_2,y_train_2)

        log = open('Classifiers/Log/LogClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_log, log)
        log.close()

        bay = open('Classifiers/Bay/BayClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_bayes, bay)
        bay.close()

        '''
        f = open('my_classifier.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
        '''

        test_image2 = np.zeros(image_arr[0].shape)
        test_image_mask2 = np.zeros(mask_arr[0].shape)
        test_img_RC2 = np.zeros(image_arr[0].shape)
        # test_img_GC2 = np.zeros(image_arr[0].shape)
        
        test_image2 = image_arr[j]
        test_image_mask2 = mask_arr[j]


        test_image_RC2 = image_arr_red_channels[j]
        test_image_GC2 = image_arr_green_channels[j]
        test_image_BC2 = image_arr_blue_channels[j]
        entropy_arr2 = entropy[j]

        intensityColumn_Arr2 = np.squeeze(test_image2.reshape([1,test_image2.shape[0]*test_image2.shape[1]])).T
        intensityColumn_Arr2 = (intensityColumn_Arr2-np.average(intensityColumn_Arr2)) / np.std(intensityColumn_Arr2)
        redChannel_Arr2 = np.squeeze(test_image_RC2.reshape([1,test_image2.shape[0]*test_image2.shape[1]])).T
        redChannel_Arr2 = ( redChannel_Arr2 - np.average(redChannel_Arr2) ) / np.std(redChannel_Arr2)
        entropy_arr = np.squeeze(entropy_arr2.reshape([1,test_image.shape[0]*test_image.shape[1]])).T

        X_val = np.column_stack((redChannel_Arr2,entropy_arr,normDistanceColumn_Arr,intensityColumn_Arr2))#,,greenChannel_Arr2))
        Y_val = np.squeeze(test_image_mask2.reshape([1,test_image_mask2.shape[0]*test_image_mask2.shape[1]])).T


        # predicts
        predictsBayes = clf_bayes.predict(X_val)
        predictsLog = clf_log.predict(X_val)

        img_pred_Log = predictsLog.reshape([test_image.shape[0],test_image.shape[1]])
        img_pred_Bayes = predictsBayes.reshape([test_image.shape[0],test_image.shape[1]])

        # Y_train_reshaped = Y_train.reshape([test_image.shape[0],test_image.shape[1]])

        #myShowImage(img_pred_Log,"img_pred_Log_" + str(j))
        #myShowImage(img_pred_Bayes,"img_pred_Bayes_" + str(j))

        try:
            coordsBBLog = newBBcoords(img_pred_Log,test_image)
        except:
            coordsBBLog = []

        try:
            coordsBBBay = newBBcoords(img_pred_Bayes,test_image)
        except:
            coordsBBBay = []

        ODROILog.append(coordsBBLog)
        ODROIBay.append(coordsBBBay)

    ODROILog_Arr = np.array(ODROILog)
    ODROIBay_Arr = np.array(ODROIBay)

    np.save('ODROILog_Arr.npy',ODROILog_Arr)
    np.save('ODROIBay_Arr.npy',ODROIBay_Arr)

prepareSegments = False

if prepareSegments:
    ODROILog_Arr = np.load('ODROILog_Arr.npy')
    ODROIBay_Arr = np.load('ODROIBay_Arr.npy')
    OD_section = []
    OD_mask = []
    OD_section_RC = []

    lenX_Arr = 0

    for i in range(0,40):
        try:
            coords = ODROILog_Arr[i]
            #myShowImage(image_arr[i][coords[0]:coords[2],coords[1]:coords[3]],"LOG" +str(i))

            segMask = np.array(mask_arr[i][coords[0]:coords[2],coords[1]:coords[3]])
            segRC = np.array(image_arr_red_channels[i][coords[0]:coords[2],coords[1]:coords[3]])

            imgSegment = np.array(image_arr[i][coords[0]:coords[2],coords[1]:coords[3]])
            vesslesSeg = np.array(vessels[i][coords[0]:coords[2],coords[1]:coords[3]])

            kernel_ones = np.ones([3,3],np.uint8)
            vesslesSeg = cv2.morphologyEx(vesslesSeg, cv2.MORPH_DILATE, kernel_ones)

            indxsVesl = np.where(vesslesSeg != 0)

            medianFiltered = median(imgSegment,disk(25))
            maxFiltered = maximum_filter(imgSegment, size=15)
            smoothVessels = np.copy(imgSegment)
            smoothVessels[indxsVesl[0],indxsVesl[1]] = np.multiply(maxFiltered[indxsVesl[0],indxsVesl[1]],0.97)
            #smoothDisk = mean(smoothVessels, disk(5)) 

            OD_section.append(smoothVessels)
            OD_mask.append(segMask)
            OD_section_RC.append(segRC)
            lenX_Arr = lenX_Arr + (imgSegment.shape[0]*imgSegment.shape[1]) 

            #coords = ODROIBay_Arr[i]
            #myShowImage(image_arr[i][coords[0]:coords[2],coords[1]:coords[3]],"BAY" + str(i))
        except:
            coords = ODROIBay_Arr[i]

            segMask = np.array(mask_arr[i][coords[0]:coords[2],coords[1]:coords[3]])
            segRC = np.array(image_arr_red_channels[i][coords[0]:coords[2],coords[1]:coords[3]])

            imgSegment = np.array(image_arr[i][coords[0]:coords[2],coords[1]:coords[3]])
            vesslesSeg = np.array(vessels[i][coords[0]:coords[2],coords[1]:coords[3]])

            kernel_ones = np.ones([3,3],np.uint8)
            vesslesSeg = cv2.morphologyEx(vesslesSeg, cv2.MORPH_DILATE, kernel_ones)

            indxsVesl = np.where(vesslesSeg != 0)

            #medianFiltered = median(imgSegment,disk(25))
            maxFiltered = maximum_filter(imgSegment, size=15)
            smoothVessels = np.copy(imgSegment)
            smoothVessels[indxsVesl[0],indxsVesl[1]] = np.multiply(maxFiltered[indxsVesl[0],indxsVesl[1]],0.97)

            #myShowImage(image_arr[i][coords[0]:coords[2],coords[1]:coords[3]],"EXCEPT" + str(i))
            OD_section.append(smoothVessels)
            OD_mask.append(segMask)
            OD_section_RC.append(segRC)
            #print('except')
            lenX_Arr = lenX_Arr + (imgSegment.shape[0]*imgSegment.shape[1]) 

        #myShowImage(smoothVessels)

    OD_section_Arr = np.array(OD_section)
    OD_mask_Arr = np.array(OD_mask)
    OD_section_RC = np.array(OD_section_RC)

    np.save('OD_section_Arr.npy',OD_section_Arr)
    np.save('OD_mask_Arr.npy',OD_mask_Arr)
    np.save('OD_section_RC.npy',OD_section_RC)

    print(lenX_Arr) # len = 4577126

finalSegmentation = False

finalMaskPredicts = []

if finalSegmentation:

    OD_section_Arr = np.load('OD_section_Arr.npy')
    OD_mask_Arr = np.load('OD_mask_Arr.npy')
    OD_section_RC = np.load('OD_section_RC.npy')

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
            
    for j in range(0,40):

        removeLen = OD_section_Arr[j].shape[0] * OD_section_Arr[j].shape[1]
        X_train = np.zeros([4577126-removeLen,2])
        Y_train = np.zeros([4577126-removeLen,1])
        
        for i in range(0,40):

            if i == j:
                continue

            test_image = OD_section_Arr[i]
            test_image_mask = OD_mask_Arr[i]
            segRC = OD_section_RC[i]
            clahePrep = np.multiply(np.copy(test_image),255)
            clahePrep = clahePrep.astype(np.uint8)
            highContrast = clahe.apply(clahePrep)

            intensityColumn_Arr = np.squeeze(test_image.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
            intensityColumn_Arr = (intensityColumn_Arr-np.average(intensityColumn_Arr)) / np.std(intensityColumn_Arr)
            segRC = np.squeeze(segRC.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
            #segRC = (segRC-np.average(segRC)) / np.std(segRC)

            if (i-1)*test_image.shape[0]*test_image.shape[1] < 0 and (i)*test_image.shape[0]*test_image.shape[1] == 0:
                X_train[(i-1)*test_image.shape[0]*test_image.shape[1]::,...] = np.column_stack((intensityColumn_Arr,segRC))#,
                Y_train[(i-1)*test_image.shape[0]*test_image.shape[1]::,0] = np.squeeze(test_image_mask.reshape([1,test_image_mask.shape[0]*test_image_mask.shape[1]])).T
                continue

            X_train[(i-1)*test_image.shape[0]*test_image.shape[1]:(i)*test_image.shape[0]*test_image.shape[1],...] = np.column_stack((intensityColumn_Arr,segRC))#,
            Y_train[(i-1)*test_image.shape[0]*test_image.shape[1]:(i)*test_image.shape[0]*test_image.shape[1],0] = np.squeeze(test_image_mask.reshape([1,test_image_mask.shape[0]*test_image_mask.shape[1]])).T

        X_train_2 = X_train
        y_train_2 = Y_train

        clf_bayes = GaussianNB()
        clf_bayes.fit(X_train_2,y_train_2)

        paramsBayes = clf_bayes.get_params

        # Logistic regression
        clf_log = LogisticRegression()
        clf_log.fit(X_train_2,y_train_2)

        log = open('Classifiers/Segments/Log/LogClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_log, log)
        log.close()

        bay = open('Classifiers/Segments/Bay/BayClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_bayes, bay)
        bay.close()

        test_image = OD_section_Arr[j]
        test_image_mask = OD_mask_Arr[j]
        segRC = OD_section_RC[j]
        clahePrep = np.multiply(np.copy(test_image),255)
        clahePrep = clahePrep.astype(np.uint8)
        highContrast = clahe.apply(clahePrep)

        intensityColumn_Arr = np.squeeze(test_image.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
        intensityColumn_Arr = (intensityColumn_Arr-np.average(intensityColumn_Arr)) / np.std(intensityColumn_Arr)
        segRC = np.squeeze(segRC.reshape([1,test_image.shape[0]*test_image.shape[1]])).T
        #segRC = (segRC-np.average(segRC)) / np.std(segRC)

        X_val = np.column_stack((intensityColumn_Arr,segRC))

        predictsBayes = clf_bayes.predict(X_val)
        predictsLog = clf_log.predict(X_val)

        img_pred_Log = predictsLog.reshape([test_image.shape[0],test_image.shape[1]])
        img_pred_Bayes = predictsBayes.reshape([test_image.shape[0],test_image.shape[1]])

        #myShowImage(img_pred_Log,"Log")
        #myShowImage(img_pred_Bayes,"Bayes")
        #myShowImage(test_image,"Actual")

        finalMaskPredicts.append(predictsBayes)

        #print('ok')

    finalMaskPredicts_Arr = np.array(finalMaskPredicts)
    np.save("finalMaskPredicts_Bayes.npy",finalMaskPredicts_Arr)

loadFinalSegs = False

if loadFinalSegs:
    foveaBBoxCoords = []
    centroidCoord = []
    ODmaskPredicts = []
    elips = np.load('elips_arr.npy')
    originalDimsBase = np.zeros(image_arr[0].shape)
    OD_section_Arr = np.load('OD_section_Arr.npy')
    finalMaskPredicts_Arr = np.load("finalMaskPredicts_Bayes.npy")
    ODROILog_Arr = np.load('ODROILog_Arr.npy')
    ODROIBay_Arr = np.load('ODROIBay_Arr.npy')
    for i in range(0,40):
        originalDims = np.copy(originalDimsBase)
        test_image = OD_section_Arr[i]
        maskPred = finalMaskPredicts_Arr[i].reshape([test_image.shape[0],test_image.shape[1]])
        finalMask, centroidCoords, radius, colsCoord = getLargestAreaEcentroid(maskPred)
        finalMaskImg = np.multiply(finalMask,255)
        finalMaskImg[centroidCoords[0],centroidCoords[1]] = 255

        try:
            coords = ODROILog_Arr[i]
            failTest = (coords[2])
        except:
            coords = ODROIBay_Arr[i]
            failTest = (coords[2])

        coordsReal =[centroidCoords[0] + coords[0],centroidCoords[1] + coords[1]] 
        colsCoordReal = [colsCoord[0] + coords[1],colsCoord[1] + coords[1]]

        originalDims[coords[0]:coords[2],coords[1]:coords[3]] = finalMaskImg
        #originalDims = originalDims or elips[i]

        elipsResized = cv2.resize(elips[i], dsize=(originalDims.shape[1],originalDims.shape[0]), interpolation=cv2.INTER_CUBIC)
        elipsResized = np.average(elipsResized,axis = 2) # 3 channels -> 1 channel
        elipsResized[elipsResized>0.5] = 1
        elipsResized[elipsResized<1] = 0

        elipsResized = thin(elipsResized)

        elipsIndexs = np.where(elipsResized != 0)


        originalDims = originalDims.astype(np.uint8)
        #originalDims[elipsIndexs] = 255
        indexsOD_ELi = np.where(originalDims != 0)
        #myShowImage(originalDims,str(i))

        checkResults = np.copy(image_arr[i])
        checkResults[indexsOD_ELi] = originalDims[indexsOD_ELi]
        #checkResults[0::,np.min(elipsIndexs[1])] = 255 # left
        #checkResults[0::,np.max(elipsIndexs[1])] = 255 # right

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

        relevantRows = np.where(elipsResized[...,relevantColumn]!=0)

        checkResults[relevantRows[0][0]:relevantRows[0][-1],columnROI_f[0]] = 0 # 1 - columnROI_f[0]
        checkResults[relevantRows[0][0]:relevantRows[0][-1],columnROI_f[1]] = 0 # 3 - columnROI_f[1]
        checkResults[relevantRows[0][0],columnROI_f[0]:columnROI_f[1]] = 0 # 0 - relevantRows[0][0]
        checkResults[relevantRows[0][-1],columnROI_f[0]:columnROI_f[1]] = 0 # 2 - relevantRows[0][-1]

        foveaBBoxCoords.append((relevantRows[0][0],columnROI_f[0],relevantRows[0][-1],columnROI_f[1]))
        centroidCoord.append(coordsReal)
        originalDims = np.divide(originalDims,255)
        ODmaskPredicts.append(originalDims)

        #myShowImage(originalDims,str(i))
        #myShowImage(checkResults,str(i))

    foveaBBoxCoords_Arr = np.array(foveaBBoxCoords)
    centroidCoord_Arr = np.array(centroidCoord)
    ODmaskPredicts_Arr = np.array(ODmaskPredicts)
    
    np.save("bbox_fovea.npy",foveaBBoxCoords_Arr)
    np.save("centroidCoord_Arr.npy",centroidCoord_Arr)
    np.save("ODmaskPredicts_Arr.npy",ODmaskPredicts_Arr)
        
getFoveaGTCoords = True

if getFoveaGTCoords:
    
    foveCoordsGT = []
    tempCoords =[]
    imgNo = 0

    with open('Datasets/fovea_location.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            #print(row)
            tempCoords.append(float(row[1]))
            tempCoords.append(float(row[2]))
            foveCoordsGT.append(tempCoords)
            tempCoords =[]
            imgNo += 1
            if imgNo == 40:
                break
    
getFoveaCoordsPred = False

'''for i in range(0,40):

    myShowImage(image_arr[i])
    myShowImage(image_arr_red_channels[i])
    myShowImage(image_arr_green_channels[i])
    myShowImage(vessels[i])
    myShowImage(entropy_arr[i])'''

if getFoveaCoordsPred:

    foveaBBoxCoords_Arr = np.load("bbox_fovea.npy")
    foveaBBoxCoords_Arr = np.absolute(foveaBBoxCoords_Arr)
    removeLen = 0
    realCentroidCoords_Arr = []
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))

    for i in range(0,40): # not the best way...

        if foveaBBoxCoords_Arr[i][3] < foveaBBoxCoords_Arr[i][1]:
            temp = foveaBBoxCoords_Arr[i][1]
            foveaBBoxCoords_Arr[i][1] = foveaBBoxCoords_Arr[i][3]
            foveaBBoxCoords_Arr[i][3] = temp
        
        if foveaBBoxCoords_Arr[i][2] < foveaBBoxCoords_Arr[i][0]:
            temp = foveaBBoxCoords_Arr[i][0]
            foveaBBoxCoords_Arr[i][0] = foveaBBoxCoords_Arr[i][2]
            foveaBBoxCoords_Arr[i][2] = temp

        test_image = image_arr[i]

        fovea_region = test_image[foveaBBoxCoords_Arr[i][0]:foveaBBoxCoords_Arr[i][2],foveaBBoxCoords_Arr[i][1]:foveaBBoxCoords_Arr[i][3]] 
        bboxShape = fovea_region.shape

        removeLen += bboxShape[0]*bboxShape[1]    

    #print(removeLen)

    for j in range(0,40):

        removeLen = (foveaBBoxCoords_Arr[j][2]-foveaBBoxCoords_Arr[j][0]) * (foveaBBoxCoords_Arr[j][3]-foveaBBoxCoords_Arr[j][1])    
        X_train = np.zeros([3187816-removeLen,3]) # 3187816 = number of points in all fovea bboxs
        Y_train = np.zeros([3187816-removeLen,1])
        
        first = 0

        for i in range(0,40):

            if i == j:
                continue

            '''if foveaBBoxCoords_Arr[i][3] < foveaBBoxCoords_Arr[i][1]:
                temp = foveaBBoxCoords_Arr[i][1]
                foveaBBoxCoords_Arr[i][1] = foveaBBoxCoords_Arr[i][3]
                foveaBBoxCoords_Arr[i][3] = temp
            
            if foveaBBoxCoords_Arr[i][2] < foveaBBoxCoords_Arr[i][0]:
                temp = foveaBBoxCoords_Arr[i][0]
                foveaBBoxCoords_Arr[i][0] = foveaBBoxCoords_Arr[i][2]
                foveaBBoxCoords_Arr[i][2] = temp'''
            
            test_image = image_arr[i]

            fovea_region = test_image[foveaBBoxCoords_Arr[i][0]:foveaBBoxCoords_Arr[i][2],foveaBBoxCoords_Arr[i][1]:foveaBBoxCoords_Arr[i][3]] 
            bboxShape = fovea_region.shape
            last = bboxShape[0]*bboxShape[1] + first
            foveaRegionGC = image_arr_green_channels[i][foveaBBoxCoords_Arr[i][0]:foveaBBoxCoords_Arr[i][2],foveaBBoxCoords_Arr[i][1]:foveaBBoxCoords_Arr[i][3]]

            clahePrep = np.multiply(np.copy(foveaRegionGC),255)
            clahePrep = clahePrep.astype(np.uint8)
            highContrast = clahe.apply(clahePrep)

            #mask
            maskBig = np.zeros(test_image.shape)
            coordsFoveaCenter = [round(foveCoordsGT[i][1]/4),round(foveCoordsGT[i][0]/4)]
            maskBig[coordsFoveaCenter[0]-10:coordsFoveaCenter[0]+10,coordsFoveaCenter[1]-10:coordsFoveaCenter[1]+10] = 1
            mask = maskBig[foveaBBoxCoords_Arr[i][0]:foveaBBoxCoords_Arr[i][2],foveaBBoxCoords_Arr[i][1]:foveaBBoxCoords_Arr[i][3]]

            fovea_region = np.squeeze(fovea_region.reshape([1,bboxShape[0]*bboxShape[1]])).T
            fovea_region = (fovea_region-np.average(fovea_region)) / np.std(fovea_region)

            foveaRegionGC = np.squeeze(foveaRegionGC.reshape([1,bboxShape[0]*bboxShape[1]])).T
            foveaRegionGC = (foveaRegionGC-np.average(foveaRegionGC)) / np.std(foveaRegionGC)

            highContrast = np.squeeze(highContrast.reshape([1,bboxShape[0]*bboxShape[1]])).T
            highContrast = (highContrast-np.average(highContrast)) / np.std(highContrast)

            '''if (i-1)*bboxShape[0]*bboxShape[1] < 0 and (i)*bboxShape[0]*bboxShape[1] == 0:
                X_train[(i-1)*bboxShape[0]*bboxShape[1]::,...] = np.column_stack((fovea_region,foveaRegionGC,highContrast))#,
                Y_train[(i-1)*bboxShape[0]*bboxShape[1]::,0] = np.squeeze(mask.reshape([1,bboxShape[0]*bboxShape[1]])).T
                continue'''

            X_train[first:last,...] = np.column_stack((fovea_region,foveaRegionGC,highContrast))#,
            Y_train[first:last,0] = np.squeeze(mask.reshape([1,bboxShape[0]*bboxShape[1]])).T

            first = last

        X_train_2 = X_train
        y_train_2 = Y_train

        clf_bayes = GaussianNB()
        clf_bayes.fit(X_train_2,y_train_2)

        paramsBayes = clf_bayes.get_params

        # Logistic regression
        clf_log = LogisticRegression()
        clf_log.fit(X_train_2,y_train_2)

        '''log = open('Classifiers/Segments/Log/LogClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_log, log)
        log.close()

        bay = open('Classifiers/Segments/Bay/BayClf_excluding_' + str(j) + '.pickle', 'wb')
        pickle.dump(clf_bayes, bay)
        bay.close()'''

        test_image = image_arr[j]

        fovea_region = test_image[foveaBBoxCoords_Arr[j][0]:foveaBBoxCoords_Arr[j][2],foveaBBoxCoords_Arr[j][1]:foveaBBoxCoords_Arr[j][3]] 
        bboxShape = fovea_region.shape
        
        foveaRegionGC = image_arr_green_channels[j][foveaBBoxCoords_Arr[j][0]:foveaBBoxCoords_Arr[j][2],foveaBBoxCoords_Arr[j][1]:foveaBBoxCoords_Arr[j][3]]

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

        predictsBayes = clf_bayes.predict(X_val)
        predictsLog = clf_log.predict(X_val)

        img_pred_Log = predictsLog.reshape(bboxShape)
        img_pred_Bayes = predictsBayes.reshape(bboxShape)

        try:
            finalMask, centroidCoords, radius, colsCoord = getLargestAreaEcentroid(img_pred_Bayes)
            if centroidCoords.size == 0:
                finalMask = np.zeros(img_pred_Bayes.shape)
                finalMask[round(finalMask.shape[0]/2),round(finalMask.shape[1]/2)] = 1
                centroidCoords = np.array([round(finalMask.shape[0]/2),round(finalMask.shape[1]/2)])

        except:
            finalMask = np.zeros(img_pred_Bayes.shape)
            finalMask[round(finalMask.shape[0]/2),round(finalMask.shape[1]/2)] = 1
            centroidCoords = np.array([round(finalMask.shape[0]/2),round(finalMask.shape[1]/2)])

        maskEyes = np.copy(finalMask)
        maskEyes = np.multiply(maskEyes,255)
        maskEyes = maskEyes.astype(np.uint8) 
        #myShowImage(test_image[foveaBBoxCoords_Arr[j][0]:foveaBBoxCoords_Arr[j][2],foveaBBoxCoords_Arr[j][1]:foveaBBoxCoords_Arr[j][3]],"fovea")
        #myShowImage(maskEyes,"Mask")
        #myShowImage(img_pred_Bayes,"Bay")     

        realCentroidCoords = [centroidCoords[0] + foveaBBoxCoords_Arr[j][0],centroidCoords[1] + foveaBBoxCoords_Arr[j][1]]

        realCentroidCoords_Arr.append(realCentroidCoords)

    realCentroidCoords_Arr = np.array(realCentroidCoords_Arr)
    np.save('fovea_centre_coords.npy',realCentroidCoords_Arr)

        

    #centroidCoord_Arr = np.load("centroidCoord_Arr.npy")
    #ODmaskPredicts_Arr = np.load("ODmaskPredicts_Arr.npy")

    #for i in range(0,40):
        




showGraphsClass= False
if showGraphsClass:

    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        return xx, yy


    def plot_contours(ax, clf, xx, yy, proba=False, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        if proba:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,-1]
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])        
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z,20, **params)
        return out

    ## import some data to play with
    #iris = datasets.load_iris()
    ## Take the first two features. We could avoid this by using a two-dim dataset
    #X = iris.data[:, :2]
    #y = iris.target

    X = X_train_2
    y = y_train_2

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    models = (clf_bayes, clf_log) #, clf_svm, clf_svm_rbf)

    # title for the plots
    titles = ('Bayes',
        'Logistic regression')
    ''' ,
    'SVC with linear kernel',
    'SVM with RBF kernel')'''

    # Set-up 2x2 grid for plotting.
    #fig, sub = 
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[0::500, 0], X[0::500, 1]
    xx, yy = make_meshgrid(X0, X1,h=0.005)

    '''_,ax_all = plt.subplots(1,2)
    ax = ax_all[1]
    plot_contours(ax, clf_bayes, xx, yy,
                    cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y[0::500], cmap=plt.cm.coolwarm, s=20)
    ax.set_xlim(X0.min(), X0.max())
    ax.set_ylim(X1.min(), X1.max())
    ax.set_xlabel('Distance')
    ax.set_ylabel('Intensity')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Bayes")

    plt.show()'''
    showPlots = False

    if showPlots:

        for clf, title in zip(models, titles):
            _,ax_all = plt.subplots(1,2)
            
        
            ax = ax_all[0]
            plot_contours(ax, clf, xx, yy, proba=True, # changed proba to probability
                        cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y[0::500], cmap=plt.cm.coolwarm, s=20)
            ax.set_xlim(X0.min(), X0.max())
            ax.set_ylim(X1.min(), X1.max())
            ax.set_xlabel('Distance')
            ax.set_ylabel('Intensity')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)

            ax = ax_all[1]
            plot_contours(ax, clf, xx, yy,
                        cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y[0::500], cmap=plt.cm.coolwarm, s=20)
            ax.set_xlim(X0.min(), X0.max())
            ax.set_ylim(X1.min(), X1.max())
            ax.set_xlabel('Distance')
            ax.set_ylabel('Intensity')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
            
        plt.show()


print("Done")