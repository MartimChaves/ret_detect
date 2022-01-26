# this needs to be in the previous directory

import cv2.cv2 as cv2
import numpy as np
import os

from skimage.measure import label, regionprops

os.chdir('C:/Users/Martim_Pc/Desktop/DACO/PROJECT_DACO/convNet/Unet')

def myShowImage(img,name = "from_show_function"):
    cv2.imshow(name, img) 

    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    return

def getBBox(img_mask):

    labels, numRegs = label(img_mask, neighbors=8, background = 0, return_num = True)
    regionsLog = regionprops(labels)

    bbxs = [regionsLog[x]['bbox'] for x in range(numRegs)]

    return bbxs


def getLargestAreaEcentroid(img_pred_Log):
    # returns mask with the regions with the largest area, coords of centroid and radius

    labelsLog, numLog = label(img_pred_Log, neighbors=8, background = 0, return_num = True)
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



testImage = [17,18,19]#20,21,22]
testRun = True
imageNumberName = 1
testImgNumber = 1



for i in range(1,41):

    if testRun:
        if i not in testImage:
            continue

    imgPath = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '.jpg' 
    imgPathOD_Masks = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '_OD.tif' 
    imgPathEX_Masks = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '_EX.tif' 

    img = cv2.imread(imgPath)#, cv2.CV_8UC1)
    Od_mask = cv2.imread(imgPathOD_Masks, cv2.CV_8UC1)
    Ex_mask = cv2.imread(imgPathEX_Masks, cv2.CV_8UC1)

    maskRed = img[...,0]>30
    maskGreen = img[...,1]>30
    maskBlue = img[...,2]>30
    mask1 = np.logical_or(maskRed,maskGreen)
    maskFinal = np.logical_or(mask1,maskBlue)
    zeros = np.zeros(img.shape)
    zeros[maskFinal] = img[maskFinal]
    zeros = zeros.astype(np.uint8)
    
    img = np.copy(zeros)
    firstCol = np.min(np.where(img!=0)[1])
    lastCol = np.max(np.where(img!=0)[1])

    #print(np.max(img))
    #myShowImage(cv2.resize(img, (416,416), interpolation = cv2.INTER_AREA))

    finalWidth = int(lastCol-firstCol)
    offsetRows = int(round((finalWidth-img.shape[0])/2))

    #Get a squared image
    #Retina image 
    squareToBePatched = np.zeros([finalWidth,finalWidth,3])
    squareToBePatched[offsetRows:img.shape[0]+offsetRows,::] = np.copy(img[::,firstCol:lastCol])

    #OD mask
    ODsquareToBePatched = np.zeros([finalWidth,finalWidth])
    ODsquareToBePatched[offsetRows:img.shape[0]+offsetRows,::] = np.copy(Od_mask[::,firstCol:lastCol])


    _, centreCoords, radius, _ = getLargestAreaEcentroid(ODsquareToBePatched)
    print(radius)

    #Ex mask
    EXsquareToBePatched = np.zeros([finalWidth,finalWidth])
    EXsquareToBePatched[offsetRows:img.shape[0]+offsetRows,::] = np.copy(Ex_mask[::,firstCol:lastCol])

    # Defining patch size
    patchSize = int(finalWidth/7)

    createGeneralPatches = False
    if createGeneralPatches: 
        if not testRun:
            if i not in testImage:

                for j in range(7):
                    for k in range(7):
                        # build patch
                        print("Creating patch: ",str(imageNumberName))
                        # select region
                        patchImg = np.copy(squareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        patchOD = np.copy(ODsquareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        patchEX = np.copy(EXsquareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        # resize to 416x416
                        patchImg = cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA)
                        patchOD = cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA)
                        patchEX = cv2.resize(patchEX, (416,416), interpolation = cv2.INTER_AREA)
                        # threshold
                        patchOD = np.multiply(patchOD,255/np.max(patchOD))
                        patchOD[patchOD>220] = 255
                        patchOD[patchOD<=220] = 0
                        patchOD = patchOD.astype(np.uint8)
                        patchEX = np.multiply(patchEX,255/np.max(patchEX))
                        patchEX[patchEX>220] = 255
                        patchEX[patchEX<=220] = 0
                        patchEX = patchEX.astype(np.uint8)
                        # if black -> remove
                        indxsBlack = np.where(patchImg < 5)
                        if len(indxsBlack[0]) > 259584:
                            continue
                        #get bboxs
                        bbxsOd = getBBox(patchOD)
                        bbxsEx = getBBox(patchEX)
                        #save img and info to annot file

                        # if a lot of stuff - data augment
                        seenClasses = False
                        if len(bbxsOd) > 0:
                            seenClasses = True
                            # do more of these
                            # 4 crops for each flip, 3 flips - 16
                            cropAssistValues = {0:[-30,-30],1:[-30,+30],2:[+30,-30],3:[+30,+30],
                                                4:[0,-30],5:[-30,0],6:[0,+30],7:[+30,0]} #cropAssistValues[c][0 ou 1]
                            for c in range(8): # crops
                                try: 
                                    patchImg = np.copy(squareToBePatched[(j*patchSize)+cropAssistValues[c][0]:((j+1)*patchSize)+cropAssistValues[c][0],
                                                                        (k*patchSize+cropAssistValues[c][1]):((k+1)*patchSize)+cropAssistValues[c][0]])  

                                    patchOD = np.copy(ODsquareToBePatched[(j*patchSize)+cropAssistValues[c][0]:((j+1)*patchSize)+cropAssistValues[c][0],
                                                                        (k*patchSize+cropAssistValues[c][1]):((k+1)*patchSize)+cropAssistValues[c][0]])  

                                    for f in range(4): # flips
                                        patchImg = np.rot90(patchImg,axes=(0,1))
                                        patchOD = np.rot90(patchOD,axes=(0,1))

                                        patchImg = cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA)
                                        patchOD = cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA)

                                        # threshold
                                        patchOD = np.multiply(patchOD,255/np.max(patchOD))
                                        patchOD[patchOD>220] = 255
                                        patchOD[patchOD<=220] = 0
                                        patchOD = patchOD.astype(np.uint8)

                                        bbxsOd = getBBox(patchOD)

                                        if len(bbxsOd) > 0: 

                                            image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/train", "%06d.jpg" %(imageNumberName)))
                                            cv2.imwrite(image_path,patchImg)
                                            annotation = image_path

                                            for k in range(len(bbxsOd)):
                                                xmin = str(bbxsOd[k][1])
                                                ymin = str(bbxsOd[k][0])
                                                xmax = str(bbxsOd[k][3])
                                                ymax = str(bbxsOd[k][2])
                                                class_ind = str(0)
                                                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                                            wf.write(annotation + "\n")
                                            imageNumberName += 1
                                except:
                                    print("Error croping.")
                                    
                        if len(bbxsEx) > 0:
                            seenClasses = True
                            # do more of these
                            # do more of these
                            # 4 crops for each flip, 3 flips - 16
                            cropAssistValues = {0:[-30,-30],1:[-30,+30],2:[+30,-30],3:[+30,+30]} #cropAssistValues[c][0 ou 1]
                            for c in range(4): # crops
                                try:
                                    patchImg = np.copy(squareToBePatched[(j*patchSize)+cropAssistValues[c][0]:((j+1)*patchSize)+cropAssistValues[c][0],
                                                                        (k*patchSize+cropAssistValues[c][1]):((k+1)*patchSize)+cropAssistValues[c][0]])  

                                    patchOD = np.copy(EXsquareToBePatched[(j*patchSize)+cropAssistValues[c][0]:((j+1)*patchSize)+cropAssistValues[c][0],
                                                                        (k*patchSize+cropAssistValues[c][1]):((k+1)*patchSize)+cropAssistValues[c][0]])  

                                    for f in range(4): # flips
                                        patchImg = np.rot90(patchImg,axes=(0,1))
                                        patchOD = np.rot90(patchOD,axes=(0,1))

                                        patchImg = cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA)
                                        patchOD = cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA)

                                        # threshold
                                        patchOD = np.multiply(patchOD,255/np.max(patchOD))
                                        patchOD[patchOD>220] = 255
                                        patchOD[patchOD<=220] = 0
                                        patchOD = patchOD.astype(np.uint8)

                                        bbxsOd = getBBox(patchOD)

                                        if len(bbxsOd) > 0:

                                            image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/train", "%06d.jpg" %(imageNumberName)))
                                            cv2.imwrite(image_path,patchImg)
                                            annotation = image_path

                                            for k in range(len(bbxsOd)):
                                                xmin = str(bbxsOd[k][1])
                                                ymin = str(bbxsOd[k][0])
                                                xmax = str(bbxsOd[k][3])
                                                ymax = str(bbxsOd[k][2])
                                                class_ind = str(1)
                                                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                                            wf.write(annotation + "\n")
                                            imageNumberName += 1
                                except: 
                                    print("Error croping.")

                        if seenClasses:
                            continue

                        image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/train", "%06d.jpg" %(imageNumberName)))
                        cv2.imwrite(image_path,patchImg)
                        annotation = image_path

                        for k in range(len(bbxsOd)):
                            xmin = str(bbxsOd[k][1])
                            ymin = str(bbxsOd[k][0])
                            xmax = str(bbxsOd[k][3])
                            ymax = str(bbxsOd[k][2])
                            class_ind = str(0)
                            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                        for k in range(len(bbxsEx)):
                            xmin = str(bbxsEx[k][1])
                            ymin = str(bbxsEx[k][0])
                            xmax = str(bbxsEx[k][3])
                            ymax = str(bbxsEx[k][2])
                            class_ind = str(1)
                            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                        wf.write(annotation + "\n")
                        imageNumberName += 1
        else:
            if i in testImage:
                for j in range(7):
                    for k in range(7):
                        #build patch
                        print("Creating patch.")
                        #select region
                        patchImg = np.copy(squareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        patchOD = np.copy(ODsquareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        patchEX = np.copy(EXsquareToBePatched[j*patchSize:(j+1)*patchSize,k*patchSize:(k+1)*patchSize])
                        # resize to 416x416
                        patchImg = cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA)
                        patchOD = cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA)
                        patchEX = cv2.resize(patchEX, (416,416), interpolation = cv2.INTER_AREA)
                        # threshold
                        patchOD = np.multiply(patchOD,255/np.max(patchOD))
                        patchOD[patchOD>220] = 255
                        patchOD[patchOD<=220] = 0
                        patchOD = patchOD.astype(np.uint8)
                        patchEX = np.multiply(patchEX,255/np.max(patchEX))
                        patchEX[patchEX>220] = 255
                        patchEX[patchEX<=220] = 0
                        patchEX = patchEX.astype(np.uint8)
                        # if black -> remove
                        indxsBlack = np.where(patchImg < 5)
                        if len(indxsBlack[0]) > 259584:
                            continue
                        #get bboxs
                        bbxsOd = getBBox(patchOD)
                        bbxsEx = getBBox(patchEX)
                        #save img and info to annot file
                        image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/test", "%06d.jpg" %(imageNumberName)))
                        cv2.imwrite(image_path,patchImg)
                        annotation = image_path

                        for k in range(len(bbxsOd)):
                            xmin = str(bbxsOd[k][1])
                            ymin = str(bbxsOd[k][0])
                            xmax = str(bbxsOd[k][3])
                            ymax = str(bbxsOd[k][2])
                            class_ind = str(0)
                            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                        for k in range(len(bbxsEx)):
                            xmin = str(bbxsEx[k][1])
                            ymin = str(bbxsEx[k][0])
                            xmax = str(bbxsEx[k][3])
                            ymax = str(bbxsEx[k][2])
                            class_ind = str(1)
                            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                        wf.write(annotation + "\n")
                        imageNumberName += 1

    createODPatches = True

    if createODPatches:

        if not testRun:
            if i not in testImage:

                cropAssistValues = {0:[0,-296],1:[0,-592],2:[0,-888],3:[0,-1184],4:[0,-1480],
                                    5:[-296,-296],6:[-296,-592],7:[-296,-888],8:[-296,-1184],9:[-296,-1480],
                                    10:[-592,-296],11:[-592,-592],12:[-592,-888],13:[-592,-1184],14:[-592,-1480],
                                    15:[-888,-296],16:[-888,-592],17:[-888,-888],18:[-888,-1184],19:[-888,-1480],
                                    20:[-1184,-296],21:[-1184,-592],22:[-1184,-888],23:[-1184,-1184],24:[-1184,-1480],
                                    25:[-1480,-296],26:[-1480,-592],27:[-1480,-888],28:[-1480,-1184],29:[-1480,-1480],
                                    30:[-148,-296],31:[-148,-592],32:[-148,-888],33:[-148,-1184],34:[-148,-1480],
                                    35:[-700,-296],36:[-700,-592],37:[-700,-888],38:[-700,-1184],39:[-700,-1480]} #cropAssistValues[c][0 ou 1]
                for w in range(40):
                    try: 
                        patchImg = np.copy(squareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])])  

                        patchOD = np.copy(ODsquareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])]) 

                        if patchImg.shape[0] != 1480 or patchImg.shape[1] != 1480:
                            continue

                        #myShowImage(cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA).astype(np.uint8))
                        
                        for f in range(4): # flips
                            patchImg = np.rot90(patchImg,axes=(0,1))
                            patchOD = np.rot90(patchOD,axes=(0,1))

                            patchImg = cv2.resize(patchImg, (256,256), interpolation = cv2.INTER_AREA)
                            patchOD = cv2.resize(patchOD, (256,256), interpolation = cv2.INTER_AREA)

                            # threshold
                            patchOD = np.multiply(patchOD,255/np.max(patchOD))
                            patchOD[patchOD>220] = 255
                            patchOD[patchOD<=220] = 0
                            patchOD = patchOD.astype(np.uint8)

                            bbxsOd = getBBox(patchOD)

                            '''image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/train", "%06d.jpg" %(imageNumberName)))
                            cv2.imwrite(image_path,patchImg.astype(np.uint8))'''

                            if len(bbxsOd) > 0: 

                                #image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/train", "%06d.jpg" %(imageNumberName)))
                                image_path = os.path.realpath(os.path.join("train/images", str(imageNumberName)+".jpg"))
                                cv2.imwrite(image_path,patchImg.astype(np.uint8))

                                label_path = os.path.realpath(os.path.join("train/labels", str(imageNumberName)+".jpg"))
                                cv2.imwrite(label_path,patchOD)
                                
                                annotation = image_path

                                '''for k in range(len(bbxsOd)):
                                    xmin = str(bbxsOd[k][1])
                                    ymin = str(bbxsOd[k][0])
                                    xmax = str(bbxsOd[k][3])
                                    ymax = str(bbxsOd[k][2])
                                    class_ind = str(0)
                                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                                wf.write(annotation + "\n")'''
                                print("Generating patch: ",str(imageNumberName))
                                imageNumberName += 1
                    except:
                        print("Failed to crop.")
        else:
            if i in testImage:

                cropAssistValues = {0:[0,-296],1:[0,-592],2:[0,-888],3:[0,-1184],4:[0,-1480],
                                    5:[-296,-296],6:[-296,-592],7:[-296,-888],8:[-296,-1184],9:[-296,-1480],
                                    10:[-592,-296],11:[-592,-592],12:[-592,-888],13:[-592,-1184],14:[-592,-1480],
                                    15:[-888,-296],16:[-888,-592],17:[-888,-888],18:[-888,-1184],19:[-888,-1480],
                                    20:[-1184,-296],21:[-1184,-592],22:[-1184,-888],23:[-1184,-1184],24:[-1184,-1480],
                                    25:[-1480,-296],26:[-1480,-592],27:[-1480,-888],28:[-1480,-1184],29:[-1480,-1480],
                                    30:[-148,-296],31:[-148,-592],32:[-148,-888],33:[-148,-1184],34:[-148,-1480],
                                    35:[-700,-296],36:[-700,-592],37:[-700,-888],38:[-700,-1184],39:[-700,-1480]} #cropAssistValues[c][0 ou 1]
                for w in range(40):
                    try: 
                        patchImg = np.copy(squareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])])  

                        patchOD = np.copy(ODsquareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])]) 

                        if patchImg.shape[0] != 1480 or patchImg.shape[1] != 1480:
                            continue

                        #myShowImage(cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA).astype(np.uint8))
                        
                        for f in range(4): # flips
                            patchImg = np.rot90(patchImg,axes=(0,1))
                            patchOD = np.rot90(patchOD,axes=(0,1))

                            patchImg = cv2.resize(patchImg, (256,256), interpolation = cv2.INTER_AREA)
                            patchOD = cv2.resize(patchOD, (256,256), interpolation = cv2.INTER_AREA)

                            # threshold
                            patchOD = np.multiply(patchOD,255/np.max(patchOD))
                            patchOD[patchOD>220] = 255
                            patchOD[patchOD<=220] = 0
                            patchOD = patchOD.astype(np.uint8)

                            bbxsOd = getBBox(patchOD)

                            if len(bbxsOd) > 0: 

                                #image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/test", "%06d.jpg" %(imageNumberName)))
                                image_path = os.path.realpath(os.path.join("test/images", str(imageNumberName)+".jpg"))
                                cv2.imwrite(image_path,patchImg.astype(np.uint8))

                                label_path = os.path.realpath(os.path.join("test/labels", str(imageNumberName)+".jpg"))
                                cv2.imwrite(label_path,patchOD)

                                annotation = image_path

                                '''for k in range(len(bbxsOd)):
                                    xmin = str(bbxsOd[k][1])
                                    ymin = str(bbxsOd[k][0])
                                    xmax = str(bbxsOd[k][3])
                                    ymax = str(bbxsOd[k][2])
                                    class_ind = str(0)
                                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                                wf.write(annotation + "\n")'''
                                print("Generating patch: ",str(imageNumberName))
                                imageNumberName += 1
                    except:
                        print("Failed to crop.")

    # Divide image in 4x3 regions
    # check if in each region there are exudates
    # if there are, get largest one and do data aug based on that one - crops and rotations
    createExPatches = False
    kernel_ones = np.ones([25,25])

    if createExPatches:
        for g in range(4):
            for t in range(3):
                #get patch:
                patchImg = np.copy(squareToBePatched[int(round(g*finalWidth/4)):int(round((g+1)*finalWidth/4)),int(round(t*finalWidth/3)):int(round((t+1)*finalWidth/3))])
                patchEX = np.copy(EXsquareToBePatched[int(round(g*finalWidth/4)):int(round((g+1)*finalWidth/4)),int(round(t*finalWidth/3)):int(round((t+1)*finalWidth/3))])

                patchEX = cv2.morphologyEx(patchEX, cv2.MORPH_CLOSE, kernel_ones)

                bbxsEx = getBBox(patchEX)

                if len(bbxsEx) > 0:
                    _, centreCoords, radius, _ = getLargestAreaEcentroid(patchEX)
                else: 
                    continue

                cropAssistValues = {0:[0,-296],1:[0,-592],2:[0,-888],3:[0,-1184],4:[0,-1480],
                                        5:[-296,-296],6:[-296,-592],7:[-296,-888],8:[-296,-1184],9:[-296,-1480],
                                        10:[-592,-296],11:[-592,-592],12:[-592,-888],13:[-592,-1184],14:[-592,-1480],
                                        15:[-888,-296],16:[-888,-592],17:[-888,-888],18:[-888,-1184],19:[-888,-1480],
                                        20:[-1184,-296],21:[-1184,-592],22:[-1184,-888],23:[-1184,-1184],24:[-1184,-1480],
                                        25:[-1480,-296],26:[-1480,-592],27:[-1480,-888],28:[-1480,-1184],29:[-1480,-1480],
                                        30:[-148,-296],31:[-148,-592],32:[-148,-888],33:[-148,-1184],34:[-148,-1480],
                                        35:[-700,-296],36:[-700,-592],37:[-700,-888],38:[-700,-1184],39:[-700,-1480]} #cropAssistValues[c][0 ou 1]

                for w in range(40):
                    try: 
                        patchImg = np.copy(squareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])])  

                        patchEX = np.copy(EXsquareToBePatched[centreCoords[0]+cropAssistValues[w][0]:centreCoords[0]+1480+cropAssistValues[w][0],
                                                            (centreCoords[1]+cropAssistValues[w][1]):(centreCoords[1]+1480+cropAssistValues[w][1])]) 

                        if patchImg.shape[0] != 1480 or patchImg.shape[1] != 1480:
                            continue

                        #myShowImage(cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA).astype(np.uint8))
                        #myShowImage(cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA).astype(np.uint8))

                        patchEX = cv2.morphologyEx(patchEX, cv2.MORPH_CLOSE, kernel_ones)         

                        for f in range(4): # flips
                            patchImg = np.rot90(patchImg,axes=(0,1))
                            patchOD = np.rot90(patchEX,axes=(0,1))

                            patchImg = cv2.resize(patchImg, (416,416), interpolation = cv2.INTER_AREA)
                            patchOD = cv2.resize(patchOD, (416,416), interpolation = cv2.INTER_AREA)

                            # threshold
                            patchOD = np.multiply(patchOD,255/np.max(patchOD))
                            patchOD[patchOD>220] = 255
                            patchOD[patchOD<=220] = 0
                            patchOD = patchOD.astype(np.uint8)

                            bbxsOd = getBBox(patchOD)

                            if len(bbxsOd) > 0: 

                                image_path = os.path.realpath(os.path.join("YOLOV3/data/dataset/test", "%06d.jpg" %(imageNumberName)))
                                cv2.imwrite(image_path,patchImg.astype(np.uint8))
                                annotation = image_path

                                for k in range(len(bbxsOd)):
                                    xmin = str(bbxsOd[k][1])
                                    ymin = str(bbxsOd[k][0])
                                    xmax = str(bbxsOd[k][3])
                                    ymax = str(bbxsOd[k][2])
                                    class_ind = str(0)
                                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

                                wf.write(annotation + "\n")
                                print("Generating patch: ",str(imageNumberName))
                                imageNumberName += 1               

                    except:
                        print("Failed to crop.")


