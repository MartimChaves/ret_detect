from main import *
from scipy.spatial.distance import cdist
from skimage import measure
import pandas as pd
import scikitplot as skplt


if __name__ == "__main__":

    getResults = False
    if getResults:
    
    # GT Fovea coords

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
        foveCoordsGT = np.array(foveCoordsGT)
        np.save("metrics/foveCoordsGT.npy",foveCoordsGT)

        # GT OD mask and GT OD centre
        mask_arr = []
        centreGT_OD_coords = []
        for i in range(1,41):
            imgPathMasks = 'Datasets/IDRID training/IDRiD_' + str(i).zfill(2) + '_OD.tif' 
            img_skio_mask = io.imread(imgPathMasks, as_gray = True)
            img_resized_mask = img_skio_mask # lazy code
            img_resized_mask[img_resized_mask == np.max(img_resized_mask)] = 1 # it has not been resized
            img_resized_mask[img_resized_mask < np.max(img_resized_mask)] = 0
            _, centreCoords, _, _ = getLargestAreaEcentroid(img_resized_mask)
            mask_arr.append(img_resized_mask)
            centreGT_OD_coords.append(centreCoords)

        mask_arr = np.array(mask_arr)
        centreOD_GT_coords_Arr = np.array(centreGT_OD_coords)
        np.save("metrics/mask_arr.npy",mask_arr)
        np.save("metrics/centreOD_GT_coords_Arr.npy",centreOD_GT_coords_Arr)

        #---

        # Fovea coords prediction
        realFoveaCoords_Arr = np.load('fovea_centre_coords.npy')
        scaledFoveaCoords_Arr = np.multiply(realFoveaCoords_Arr,4)
        np.save("metrics/scaledFoveaCoords_Arr.npy",scaledFoveaCoords_Arr)

        # OD mask prediction
        ODmaskPredicts_Arr = np.load("ODmaskPredicts_Arr.npy")
        ScaledODmaskPredicts_Arr = []
        for i in range(0,40):
            scaledPred = cv2.resize(ODmaskPredicts_Arr[i], dsize=(mask_arr[0].shape[1],mask_arr[0].shape[0]), interpolation=cv2.INTER_CUBIC)
            scaledPred = np.divide(scaledPred,np.max(scaledPred))
            scaledPred[scaledPred>0.5] = 1
            scaledPred[scaledPred<1] = 0
            ScaledODmaskPredicts_Arr.append(scaledPred)

        ScaledODmaskPredicts_Arr = np.array(ScaledODmaskPredicts_Arr)
        np.save("metrics/ScaledODmaskPredicts_Arr.npy",ScaledODmaskPredicts_Arr)

        # OD centre prediction
        ODcentroidCoord_Arr = np.load("centroidCoord_Arr.npy")
        scaledODCentroidCoords_Arr = np.multiply(ODcentroidCoord_Arr,4)
        np.save("metrics/scaledODCentroidCoords_Arr.npy",scaledODCentroidCoords_Arr)

    #---

    # TP/FP and TN/FN for OD mask
    extractMetrics = False

    if extractMetrics:
        if not getResults:
            # Load ground da trudes
            foveCoordsGT = np.load("metrics/foveCoordsGT.npy")
            mask_arr = np.load("metrics/mask_arr.npy")
            centreOD_GT_coords_Arr = np.load("metrics/centreOD_GT_coords_Arr.npy",centreOD_GT_coords_Arr)

            # Load da predios
            scaledFoveaCoords_Arr = np.load("metrics/scaledFoveaCoords_Arr.npy")
            ScaledODmaskPredicts_Arr = np.load("metrics/ScaledODmaskPredicts_Arr.npy")
            scaledODCentroidCoords_Arr = np.load("metrics/scaledODCentroidCoords_Arr.npy")

        TruePos = []
        TrueNeg = []
        FalsePos = []
        FalseNeg = []
        Acc = []
        sensi = [] # sensitiviy
        speci = [] # specificity
        for i in range(0,40):
            valPredictTest = ScaledODmaskPredicts_Arr[i]
            valLabelTest = mask_arr[i]

            totalCount = valPredictTest.shape[0]*valPredictTest.shape[1]

            TruePositives = np.logical_and(valPredictTest, valLabelTest).sum()
            FalsePositives = valPredictTest.sum() - TruePositives
            FalseNegatives = valLabelTest.sum() - TruePositives
            TrueNegatives = np.logical_and(1-valPredictTest,1-valLabelTest).sum()

            AccuracySingle = (TruePositives + TrueNegatives) / totalCount
            sensitivitySingle = TruePositives / (TruePositives + FalseNegatives) 
            specificitySingle = TrueNegatives / (TrueNegatives + FalsePositives)

            TruePos.append(TruePositives)
            TrueNeg.append(TrueNegatives)
            FalsePos.append(FalsePositives)
            FalseNeg.append(FalseNegatives)
            Acc.append(AccuracySingle)
            sensi.append(sensitivitySingle) # sensitiviy
            speci.append(specificitySingle) # specificity

        # Mean and Std of distances between fovea centres and OD centres GT
        distFovea = [] # euclidian
        distOD_centre = []
        for i in range(0,40):
            ODpredCentre = scaledODCentroidCoords_Arr[i]
            ODrealCentre = centreOD_GT_coords_Arr[i]

            FoveaPredCentre = scaledFoveaCoords_Arr[i]
            FoveaRealCentre = foveCoordsGT[i]

            distOD = np.sqrt(np.add(np.power(ODpredCentre[0]-ODrealCentre[0],2),np.power(ODpredCentre[1]-ODrealCentre[1],2)))
            distFv = np.sqrt(np.add(np.power(FoveaPredCentre[0]-FoveaRealCentre[1],2),np.power(FoveaPredCentre[1]-FoveaRealCentre[0],2)))

            distFovea.append(distFv)
            distOD_centre.append(distOD)

            #print("distOD: ", distOD)
            #print("distFv: ", distFv)
    
        # DICE and MAD and HD for OD mask

        diceValuesArr = []
        HValuesArr = []
        MADValuesArr = []
        for i in range(0,40):
            valPredictTest = ScaledODmaskPredicts_Arr[i]
            valLabelTest = mask_arr[i]

            diceValue = 2. * (np.logical_and(valPredictTest, valLabelTest)).sum() / (valPredictTest.sum() + valLabelTest.sum())
            diceValuesArr.append(diceValue)

            contours1 = measure.find_contours(valPredictTest, 0.5)
            contours2 = measure.find_contours(valLabelTest, 0.5)

            max1 = 0
            for n in range(len(contours1)):
                if len(contours1[n]) > len(contours1[max1]):
                    max1 = n

            max2 = 0
            for n in range(len(contours2)):
                if len(contours2[n]) > len(contours2[max2]):
                    max2 = n

            distM = cdist(contours1[max1], contours2[max2], 'euclidean')  # Ma by Mb distances matrix

            colMins = distM.min(axis=0)
            rowMins = distM.min(axis=1)

            # HAUSDORFF
            HAUSDORFF = max(np.max(rowMins), np.max(colMins))
            HValuesArr.append(HAUSDORFF)

            # MAD
            colMeans = np.mean(colMins)
            rowMeans = np.mean(rowMins)
            MAD = (colMeans + rowMeans) / 2
            MADValuesArr.append(MAD)

        #print("Dice: ", diceValuesArr)
        #print("HD: ", HValuesArr)
        #print("MAD: ", MADValuesArr)

        df = pd.DataFrame({'Dice': diceValuesArr, 'HD': HValuesArr, 'MAD': MADValuesArr, 'DistFovea': distFovea, 'DistOD': distOD_centre, 'Accuracy': Acc, 'TruePos': TruePos, 'TrueNeg': TrueNeg, 'falsePos': FalsePos, 'falseNEg': FalseNeg, 'Sensitivity': sensi, 'Specificity': speci})
        df.to_csv('metrics/complete_metrics.csv')

    '''df = pd.read_csv('metrics/complete_metrics.csv')

    dice = np.array(df['Dice'])
    hausd = np.array(df['HD'])
    mad = np.array(df['MAD'])
    sensitivity = np.array(df['Sensitivity'])
    specificity = np.array(df['Specificity'])
    distFovea = np.array(df['DistFovea'])
    distODcentre = np.array(df['DistOD'])

    # df.to_excel('metrics/metrics.xlsx')
    x = 1-specificity
    y = sensitivity
    plt.plot(x,y,'o')
    plt.xlabel('1 - specificity')
    plt.ylabel('Sensitivity')
    plt.show()'''
    



    


    

            

    
    
    

