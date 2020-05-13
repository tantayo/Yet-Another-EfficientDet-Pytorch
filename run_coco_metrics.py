import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob, os
import csv
import cv2
import argparse
import yaml

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

parser = argparse.ArgumentParser('Neurala coco metrics Pytorchparser')
parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
parser.add_argument('-t', '--target_iou', type=float, default=0.75, help='Target IOU')
parser.add_argument('-m', '--max_dets', type=int, default=1024, help='Max number of detections')
args = parser.parse_args()

params = Params(f'projects/{args.project}.yml')

images_path = params.val_data_path + 'images/'
labels_path = params.val_data_path + 'labels/'

if isinstance(params.preds_path, type(None)):
    preds_path = params.val_data_path + 'neupreds/'
else:
    preds_path = params.preds_path

cat_id = params.obj_list
targetIOU = args.target_iou

def clipCordinate(coordinate):
    coordinate = np.array(coordinate)
    coordinate[coordinate < 0] = 0
    coordinate[coordinate > 1] = 1
    return coordinate

def calIOU(groundTruth, predictionBoxes):
    xGMin, yGMin, xGMax, yGMax = clipCordinate(groundTruth[0:4])
    xPMin, yPMin, xPMax, yPMax = clipCordinate(predictionBoxes[0:4])
    if (xGMin > xGMax or yGMin > yGMax):
        raise AssertionError(
            'Ground Truth Box is Malformed(XGmin > xGMax) or yGMin > yGMax: {}'.format(groundTruth)
        )
    if (xPMin > xPMax or yGMin > yGMax):
        raise AssertionError(
            "Prediction Box is Malformed(XPmin > xPMax) or yPMin > yPMax: {}".format(predictionBoxes)
        )
    #IOU is zero because of those condition
    if (xGMax < xPMin or xGMin > xPMax or yGMin > yPMax or yGMax < yPMin ): 
        return 0
    intersectionArea = (min(xGMax,xPMax) - max(xPMin,xGMin)) * (min(yGMax,yPMax) - max(yPMin,yGMin)) 
    PredictionArea = (xPMax - xPMin) * (yPMax - yPMin)
    GroundTruthArea = (xGMax - xGMin) * (yGMax - yGMin)
    return intersectionArea / (PredictionArea + GroundTruthArea - intersectionArea)


def calculateHitOrMiss(GroundTruthBoxes, PredictionBoxes, Threshold, predicted_class_score):
    #Loop through ground truth and prediction boxes .If the Prediction boxes and the ground truth overlapped by
    #a certain threshold, add the ground boxes and predboxes to list.
    #This rule applied by the PASCAL VOC 2012 metric:
    #"e.g. 5 detections (TP) of a single object is counted as 1 correct detection and 4 false detections”
    #In Order to solve the problem first loop through sorted hit_list, and mark the one that have the highest IOU
    #predicted box as TP and marked every other predicted boxes that shares the same ground truth as FP 

    hit_test = []
    hit = {}
    groundTruth = {}
    predictionTruth = {}
    pred_boxes = []
    for id, gIDs in enumerate(GroundTruthBoxes):
        groundTruth[id] = gIDs
    for id, pIDs in enumerate(PredictionBoxes):
        predictionTruth[id] = pIDs
        hit[id] = -1

    #Loop through the prediction box and ground box and append pred_key (predicted boxes unique ID for one single
    #image) Ground_key and the predicted and ground truth Box IOU
    
    #This loop creates hit_test, which is a list of predictions and groundtruths where they overlap by more than
    #the IOU threshold. It also populates hit with all -1's.
    #This loop is important for the next loop, which sets indices of hit to 1 for single entries and entires where
    #the GT isn't repeated. That assigns each GT a proposal (and so we don't have multiple proposals counting as 
    #TPs for one GT)

    for ground_key, groundBoxes in groundTruth.items():
        for pred_key, predBoxes in predictionTruth.items():
            iou = calIOU(groundBoxes[:-1], predBoxes[:-1])
            if iou >= Threshold:
                hit_test.append((pred_key, ground_key, iou))

    #Sort the boxes in the order of Ground Key and IOU ThresHold. If GroundKey are equal then sort it based on
    #the IOU Thres hold
    #eg. [(0, 0, 0.755319874047164), (1, 2, 0.5317425350002831), (6, 3, 0.7414358721002615)]
    #return
    #[(0, 0, 0.75531987) (1, 2, 0.53174254) (6, 3, 0.74143587)]
    
    #We want to set `hit[index]=1` for the box that has the highest IOU for each GT.
    #Since we sorted hit_test, we can just walk through the sorted hit_test, and since it's been sorted in
    #ascending order on GT primarily, and IOU secondarily, when the next index changes GT, the current index is the
    #highest IOU for that GT. So set hit for that index to 1.
    
    #If we're at the end of the list we know we have the highest IOU for the current GT (because there are no other
    #proposals for this GT left) so we can set hit for that index to 1.
    
    #hit for all indices is initialized to -1.

    dtype = [('key', int), ('key2', int), ('iou', float)]
    x = np.sort(np.array(hit_test,dtype), order=['key2', 'iou'])
    for id, boxes in enumerate(x):
        # x is sorted, so if the next box is a different GT from the current one, we got to the last proposal for
        #  a GT, and the last proposal for a GT is the higest IOU (because it's been sorted)
        try:
            if x[id+1][1] != x[id][1]:
                hit[boxes[0]] = 1
        # if we can't access the next index (id+1) that means we're at the end of the list, in which case we have
        # the highest IOU for the current GT
        except:
            hit[boxes[0]] = 1

    #Make the hit boxes with key prediction boxes ID and value as hit or miss.
    #1 means hit
    #0 means miss
    #return the list of format [(hit or miss),Confidencescore]
    
    #Here we've set hit(index) to 1 for TP proposals that hit a GT above the IOU threhsold, and -1 for indices that
    #either don't have a GT associated with them, or that do but another higher IOU is associated with it.

    for id, boxes in enumerate(predictionTruth):
        if hit[id] == 1:
            pred_boxes.append((1, predicted_class_score[id]))
        else:
            pred_boxes.append((0, predicted_class_score[id]))
    return sorted(pred_boxes ,key = lambda x: x[1])

def hitOrMissImages(groundImages, predImages, Threshold):
    #First filter out the score from the groundImages and PredImages then put it in the hitormiss classifier
    #and calculate the hit and miss of all the Images

    assert(len(groundImages) == len(predImages))
    result=[]
    for groundBoxes, predBoxes in zip(groundImages, predImages):
        pred_Score =[score[4] for score in predBoxes]     #This is where the prediction box score are
        result.extend(calculateHitOrMiss(groundBoxes, predBoxes, Threshold, pred_Score))
    return sorted(result, key =lambda x: x[1])[::-1]

#Calculate the PRC Curve here.

# We want to take the two lists of values,
# and set the values of the precision to the largest value to the right of it
def interpolatePrecisionValue(prec, rec):
    # So we'll reverse index the two arrays (both of the same size)
    precisionMax = -1;
    for ii in range(len(prec)-1, -1, -1):
        # Get the max precision so far
        precisionMax = max(precisionMax, prec[ii])
        # If the current value is not the max, increase it to the max so far (this is the interpolate part)
        if prec[ii] < precisionMax:
            prec[ii] = precisionMax
    return(prec, rec)

#This the same as the 11 point interpolation except that instead of taking the max precision value to the
#right at 11 points value of recall from [0.0 to 1.0] , you take the max precision at every value of recall

def AllPointInterpolatePrecisionValue(prec, rec):
    assert(len(prec) == len(rec))
    for id, recall in enumerate(rec):
        try:
            prec[id] = max(prec[id:])
        except:
            pass
    return prec, rec

def divide(TP, total):
    if (total <= 0):
        return 0
    return TP/total

def CalculatePrecisionRecall(Table, GroundTruth, maxDets):
    precision = []
    recall = []
    TP = 0
    total = 0
    # initialize lastScore to be 100% confidence
    lastScore = 1
    
    ## This line is if you want to look at the table of the PRC Curve
    length = len(Table)
    for id, (truth, score) in enumerate(Table):

        #Now calculate the PRC Curve and mAP on the number of maxDets allowed
        
        # if we've reached the maximum number of detections, end it
        total += 1
        #if id > maxDets:
        #    break
        ## check if it has reach the end of the list     
        if(id + 1 < length):
            if(Table[id][0] == 1):
                TP += 1
                #check if the current score and the next score are the same
                #if they are not that means we are at the end of the confidence score bin.
            if(Table[id + 1][1] != Table[id][1]):
                if id > 1 and len(recall) > 0:
                    if (Table[id - 1][1] == Table[id][1]):
                        precision.append(divide(TP, total))
                        recall.append(recall[-1])
                precision.append(divide(TP, total))
                recall.append(divide(TP, GroundTruth))    
        else:
            ##if so calculate recall and precision
            if(Table[id][0] == 1):
                TP += 1
            precision.append(divide(TP, total))
            recall.append(divide(TP, GroundTruth))
    return np.array(precision), np.array(recall), TP/GroundTruth, TP/total

    #GroundBox format should be xmin ,ymin,xmax,ymax
    #predictionBoxes format should be  xmin, ymin,xmax,ymax,score

def mAPSingleClass11pointInterpolation(groundBox, predictionBoxes, IOUThreshold, noOfGroundTruth, PRCurve, maxDets):
    imageResult = hitOrMissImages(groundBox, predictionBoxes, IOUThreshold)
    prec, rec = CalculatePrecisionRecall(imageResult, noOfGroundTruth, maxDets)
    mAP = 0
    PRCurve.append(interpolatePrecisionValue(prec, rec))
    for x in np.linspace(0, 1.0, num = 11):
        IP_point = np.where(rec >= x + 0.1)
        line_spacing = 0.1          

        #This here done the mAP interpolation.
        #It always get max precision to the right of the curve. 
        #example: at recall 0.3. it will look from 0.3 to 1 and get the max precision.
        #Generally , I replace each precision value with the maximum precision value to the
        #right of that recall level

        if(IP_point[0].size <= 0):
            mAP += 0
        else:
            mAP += np.max(prec[IP_point])
    return 1/10 * mAP


def mAPSingleClassAllpointInterpolation(groundBox, predictionBoxes, IOUThreshold, noOfGroundTruth, PRCurve, maxDets):
    imageResult = hitOrMissImages(groundBox, predictionBoxes, IOUThreshold)
    prec, rec, precision, recall = CalculatePrecisionRecall(imageResult, noOfGroundTruth, maxDets)
    mAP = 0
    prec, rec = AllPointInterpolatePrecisionValue(prec, rec)
    PRCurve.append((rec, prec))
    # calculate the area under the curve by adding up rectangles between points in the graph
    for id, value in enumerate(prec):
        # if the width is greater than 0, add it to the mAP
        try:
            mAP += (rec[id + 1] - rec[id]) * prec[id + 1]
        # if there is no second point to calculate the width, stop calculating the mAP
        except:
            mAP += 0
    try:
        mAP += rec[0] * prec[0]
        return mAP, precision, recall ##Get the recall and the precision of each classs
    except:
        return mAP, 0, 0

def PASCAL_VOC_MAP(groundTruth, predictionBoxes, IOUThreshold, Interpolation, maxDets):
    PRCurve = []
    rec = 0
    prec = 0
    noOfGroundTruth = 0
    for boxes in groundTruth:
        for co in boxes:
            noOfGroundTruth += 1
    if (noOfGroundTruth <= 0):
        return -1, -1, -1, [PRCurve]
    if Interpolation == 'AllPoint':
        mAP, rec, prec = mAPSingleClassAllpointInterpolation(groundTruth, predictionBoxes, IOUThreshold, noOfGroundTruth, PRCurve, maxDets)
    else:
        mAP = mAPSingleClass11pointInterpolation(groundTruth, predictionBoxes, IOUThreshold, noOfGroundTruth, PRCurve, maxDets)
        
    #Uncomment the code below to plot the PRC curve for each and for each IOU ThresHold
    #Plot the interpolated PR CUrve here
    return mAP*100, rec, prec, PRCurve

#You regroup the image into the format of 
#[[image]]
#Inside the [image]
#[[bounding Box coordiante, score ], [[bounding Box coordiante, score ]] 
#This also regroup them by catergory So ground[0] will refer to the one category,  ground[1] will refer to 
#another category

def regroup_images(groundImages, predImages):
    ground = []
    pred = []
    groundKeys = list(groundImages.keys())
    predKeys = list(predImages.keys())
    groundKeys.extend(predKeys)
    keys = set(groundKeys)
    for x in keys:
        if x in groundImages:
            ground.append(groundImages[x])
        else:
            ground.append([])
        if x in predImages:
            pred.append(predImages[x])
        else:
            pred.append([])
    return (ground, pred)

def mAPAllClass(cat_id, groundTruth, prediction, IOU, Interpolationtype, maxDets):
    recall = []
    precision = []
    mAP = []
    groundClassDict = {}
    precision = []
    PRCurve = []
    predClassDict = {}
    for id in cat_id:
        groundClassDict[id] = {}
        predClassDict[id] = {}
        
    #First go trhough each groundTruth and make dictinary data structure. Each dictionary data struture is in the
    #format Catergory ID -> Image ID -> then the image information of category Id and Image ID together 
    #This is so that I will know for that particular category ID, at which image ID the catergoty is there and 
    #the category bounding box and information

    for images in groundTruth:
        for oneImage in images:
            try:
                #Regroup the images by catergory and in which image id is the category there
                groundClassDict[oneImage[-1]][oneImage[-2]].append(oneImage)
            except:
                groundClassDict[oneImage[-1]][oneImage[-2]] = [oneImage]
                
    for images in prediction:
        for oneImage in images:
            try:
                predClassDict[oneImage[-1]][oneImage[-2]].append(oneImage)
            except:
                predClassDict[oneImage[-1]][oneImage[-2]] = [oneImage]
                
    #Calculate mAP for each class here. Going throught by category
    count = 0
    removedClass = []
    for ground, prediction in zip(groundClassDict.values(), predClassDict.values()):
        groundTruth, predictionBoxes = regroup_images(ground, prediction)
        meanAveragePrecision, rec, prec, precisonRecallCurve = PASCAL_VOC_MAP(groundTruth, predictionBoxes, IOU, Interpolationtype, maxDets)
        if(meanAveragePrecision >=0):
            recall.append(rec)
            precision.append(prec)
            PRCurve.append(precisonRecallCurve)
            mAP.append(meanAveragePrecision)
        else:
            removedClass.append(cat_id[count])
        count+=1
    return mAP,recall,precision,PRCurve,removedClass

def cocomAPAllClass(cat_id, groundTruth, prediction, IOU, targetIOU, interpolationType='AllPoint', maxDets=100):
    mAP = []
    avg_mAP = []
    recall = []
    precision = []
    PRCurve = []
    removedClass = []
    print("Starting Calculation for COCO metrics-----------------------------------------")

    #Calculate the mAp @ Different IOU 
    for iou in IOU:
        meanAP, rec, prec, precisionRecallCurve, removedClass = mAPAllClass(cat_id, groundTruth, prediction,iou,interpolationType,maxDets)
        recall.append(rec)
        precision.append(prec)
        mAP.append(meanAP) 
        PRCurve.append(precisionRecallCurve)
    for classes in removedClass:
        cat_id.remove(classes)
    for meanAP,iou in zip(mAP,IOU):
        avgmAP = sum(meanAP)/(len(cat_id))
        avg_mAP.append(avgmAP)
        print("Mean Average Precision  @ [ IOU ={}".format(iou)+ " | area = all | maxDets = {} ]  = ".format(maxDets) + str(avgmAP))
        
    print("Mean Average Precision  @ [ IOU =0.5:0.95"+  " | area = all | maxDets = {} ]  = ".format(maxDets) + str(sum(avg_mAP) / len(avg_mAP))  )
    print("Mean Average Precision  @ [ IOU ={}".format(targetIOU) +  " | area = all | maxDets = {} ] ".format(maxDets) + " for Each Class")
    horizontalBar=[]
    index = int((targetIOU / 5) * 100 % 10)
    for id,value in enumerate(mAP[index]):
        print(cat_id[id] +"\t\t{}".format(value))
        horizontalBar.append((cat_id[id], value))
    print("Avg Recall and Precision at the target IOU ----------------------------------")
    print("Average Recall  @ [ IOU ={}".format(targetIOU) +    " | area = all | maxDets = {} ]  = ".format(maxDets) + str( (sum(recall[index])/len(recall[index]))*100  ))
    print("Average Precision  @ [ IOU ={}".format(targetIOU) + " | area = all | maxDets = {} ]  = ".format(maxDets) + str( (sum(precision[index])/len(precision[index]) ) * 100) )
    print("Started Drawing graph for the PRC curve at the target IOU--------------------")
    
    #Draw the PRC Curve at the target IOU
    plt.figure(figsize=(20, 20))
    for id, curve in enumerate(PRCurve[index]):
        plt.plot(curve[0][0], curve[0][1], label = str(targetIOU) + "--" + cat_id[id], linewidth = 5.5)
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    axes = plt.axes()
    axes.set_ylim([0, 1])
    axes.set_xlim([0, 1])
    plt.legend(fontsize = 16,loc = "upper right")
    plt.title("Precision - Recall Curve for each class @{}".format(targetIOU) + "for each class", fontsize = 20)
#     plt.savefig("PRC.png")
    plt.show()
        
    #Draw the horizontal bar chart of mAP sorted order @ target IOU
    horizontalBar=sorted(horizontalBar, key = lambda x: x[1])
    y=[x[1] for x in horizontalBar]
    x=[x[0] for x in horizontalBar]
    plt.figure(figsize = (50, 50))
    plt.barh(x, y)
    plt.yticks(fontsize = 40)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize = 40)
    plt.title("Horizontal Bar Chart of mAP in the sorted Order @ {}".format(targetIOU), fontsize = 70)
#     plt.savefig("mAP.png")
    plt.show()

groundTruth = []
prediction = []
tempLabel = []
tempPred = []

image_names = []

IOU = [0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for ext in ('*.gif', '*.png', '*.jpg', '*.PNG', '*.JPG', '*.bmp', '*.BMP'):
    image_list = glob.glob(images_path + ext)
    for image in image_list:
        image_names.append(os.path.basename(image))

img_id = 0
for name in image_names:
    label = labels_path + name[0:-3] + 'txt'
    preds = preds_path + name[0:-3] + 'txt'
    image = images_path + name
    
    img = cv2.imread(image)
    r, c, z = img.shape
    
    img_id += 1
    
    with open(label) as f:
        f_csv = csv.reader(f, delimiter = ' ')
        
        for row in f_csv:
            x1, y1, x2, y2 = map(float, row[4:8])
            tempLabel.append([x1/c, y1/r, x2/c, y2/r, img_id, row[0]])
            
    with open(preds) as g:
        f_csv = csv.reader(g, delimiter = ' ')
        
        for row in f_csv:
            x1, y1, x2, y2 = map(float, row[4:8])
            conf = float(row[1])
            tempPred.append([x1/c, y1/r, x2/c, y2/r, conf, img_id, row[0]])
            
groundTruth.append(tempLabel)
prediction.append(tempPred)

cocomAPAllClass(cat_id, groundTruth, prediction, IOU, targetIOU, maxDets=args.max_dets)