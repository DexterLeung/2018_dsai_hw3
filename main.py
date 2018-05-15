import random
import math
import numpy as np
import csv

random.seed();
trainCount = 20000;
valInTrainCount = 2000;
validationProp = 0.1;
testCount = 60000;
toDigitSize = [3,4,5,6];
ALLRULES = [["-"], ["-","+"], ["*"]];
ALLRULESID = ["minus","minusSum","multi"];
toTargetSize = [[x,x+1,x*2] for x in toDigitSize];
datasets = [[[] for y in ALLRULES] for x in toDigitSize];
dictSizes = [[[] for y in ALLRULES] for x in toDigitSize];
USEOLDDATA = False;

def prepareData(digitSizeID, toRuleID, useOldData):
    DIGITSIZE = toDigitSize[digitSizeID];
    INTEND = 10**DIGITSIZE-1;
    datasetID = str(DIGITSIZE) + "-" + ALLRULESID[toRuleID];
    QUERYLEN = DIGITSIZE+1+DIGITSIZE;
    TARLEN = toTargetSize[digitSizeID][toRuleID];
    
    if (useOldData):
        print("getting old data - "+datasetID)
        with open("trainingCorpus - "+datasetID+".csv", newline='', encoding='utf-8') as csvfile:
            rd = csv.reader(csvfile);
            oldTrainData = [*rd];
        with open("validationCorpus - "+datasetID+".csv", newline='', encoding='utf-8') as csvfile:
            rd = csv.reader(csvfile);
            oldValData = [*rd];
        with open("testCorpus - "+datasetID+".csv", newline='', encoding='utf-8') as csvfile:
            rd = csv.reader(csvfile);
            oldTestData = [*rd];
        conv = [[*r[0], *r[1]] for r in [*oldTrainData[1:], *oldValData[1:], *oldTestData[1:]]];
        
    else:
        def calcTarget(a,b,op):
            if op == "+":
                return a+b;
            elif op == "-":
                return a-b;
            elif op == "*":
                return a*b;
            elif op == "/":
                return a//b;

        def padEnd(string, length, pad=" "):
            while (len(string) < length):
                string += " ";
            return string;

        def createDataSource(count=1000):
            ary = [["Formula", "Answer"]];
            for i in range(0, count):
                a = math.floor(random.random()*INTEND);
                b = math.floor(random.random()*INTEND);
                if (a<b):
                    a,b = b,a;
                op = ALLRULES[toRuleID][math.floor(random.random()*len(ALLRULES[toRuleID]))];
                target = calcTarget(a,b,op);
                ary.append([padEnd(str(a)+op+str(b), QUERYLEN), padEnd(str(target),TARLEN)]);
            return ary;


        print("creating source - "+datasetID)
        source = createDataSource(trainCount+testCount);
        header = source[0];
        trainData = source[1:trainCount-valInTrainCount];
        validationData = source[trainCount-valInTrainCount:trainCount];
        testData = source[trainCount:];
        with open("trainingCorpus - "+datasetID+".csv", 'w', newline='', encoding='utf-8') as csvfile:
            toWriter = csv.writer(csvfile);
            toWriter.writerow(header);
            for r in trainData:
                toWriter.writerow(r);
        with open("validationCorpus - "+datasetID+".csv", 'w', newline='', encoding='utf-8') as csvfile:
            toWriter = csv.writer(csvfile);
            toWriter.writerow(header);
            for r in validationData:
                toWriter.writerow(r);
        with open("testCorpus - "+datasetID+".csv", 'w', newline='', encoding='utf-8') as csvfile:
            toWriter = csv.writer(csvfile);
            toWriter.writerow(header);
            for r in validationData:
                toWriter.writerow(r);

        print("preparing structire-1 - "+datasetID)
        conv = [[*r[0], *r[1]] for r in source][1:];

    print("creating one hot encoding - "+datasetID)
    oneHotPool = set();
    for r in conv:
        for i in r:
            oneHotPool.add(i);
    oneHotDict = [*oneHotPool];
    dictSizes[digitSizeID][toRuleID] = len(oneHotDict);
    toMap = {f: t for t,f in enumerate(oneHotDict)}
    fromMap = {t: f for t,f in enumerate(oneHotDict)}

    print("preparing structire-2 - "+datasetID)
    conv2 = np.asarray([[[True if c == val else False for en,val in fromMap.items()] for idx,c in enumerate(r)] for r in conv]);
    allTrainData = conv2[:trainCount];
    allTestData = conv2[trainCount:];
    trainData = allTrainData[:-valInTrainCount];
    valData = allTrainData[-valInTrainCount:];
    datasets[digitSizeID][toRuleID] = {"train": trainData, "valid": valData, "test": allTestData, "oneHotMap": fromMap};



for i in range(0,len(toDigitSize)):
    for j in range(0,len(ALLRULES)):
        prepareData(i, j, USEOLDDATA);



print("preparing models")



from keras.models import Sequential
from keras import layers
import numpy as np


def trainModel(digitSizeID=0, toRuleID=0, layerCount=1, trainingSize=1, hiddenSize=128, epochSize = 100, modelID=0):
    allData = datasets[digitSizeID][toRuleID];
    DIGITS = toDigitSize[digitSizeID];
    TARGETSIZE = toTargetSize[digitSizeID][toRuleID];
    QUERYLEN = DIGITS + 1 + DIGITS;
    RNN = layers.LSTM;
    HIDDEN_SIZE = hiddenSize;
    BATCH_SIZE = 128;
    DICT_SIZE = dictSizes[digitSizeID][toRuleID];

    print('Build model...')
    if modelID == 0:
        model = Sequential()
        model.add(RNN(HIDDEN_SIZE, input_shape=(QUERYLEN, DICT_SIZE)))
        model.add(layers.RepeatVector(TARGETSIZE))
        for i in range(0,layerCount):
            model.add(RNN(HIDDEN_SIZE, return_sequences=True))

        model.add(layers.TimeDistributed(layers.Dense(DICT_SIZE)))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        model.summary()
    elif modelID == 1:
        model = Sequential()
        model.add(RNN(HIDDEN_SIZE, input_shape=(QUERYLEN, DICT_SIZE), return_sequences=True))
        model.add(layers.Reshape((HIDDEN_SIZE, QUERYLEN)))
        model.add(layers.TimeDistributed(layers.Dense(TARGETSIZE)))
        model.add(layers.Reshape((TARGETSIZE, HIDDEN_SIZE)))
        for i in range(0,layerCount):
            model.add(RNN(HIDDEN_SIZE, return_sequences=True))

        model.add(layers.TimeDistributed(layers.Dense(DICT_SIZE)))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        model.summary()

    csvLog = [];
    finalLoss = 0;
    finalAccuracy = 0;
    finalValAccuracy = 0;
    configInfo = [toDigitSize[digitSizeID], ALLRULESID[toRuleID], layerCount, trainingSize, hiddenSize, epochSize, modelID];

    trainingDataSubset = allData["train"][:math.floor(trainingSize*trainingSize)];
    for i in range(0,epochSize):
        print('=' * 50)
        print('Iteration', i)
        history = model.fit(allData["train"][:,:QUERYLEN], allData["train"][:,QUERYLEN:],
                batch_size=BATCH_SIZE,
                epochs=1,
                validation_data=(allData["valid"][:,:QUERYLEN], allData["valid"][:,QUERYLEN:]));
        finalLoss, finalAccuracy, finalValAccuracy =history.history["loss"], history.history["acc"], history.history["val_acc"];
        csvLog.append([*configInfo, history.history["loss"], history.history["acc"], history.history["val_acc"]]);
    
    with open("trainingLog.csv", 'a', newline='', encoding='utf-8') as csvfile:
        toWriter = csv.writer(csvfile);
        for r in csvLog:
            toWriter.writerow(r);

    testCorrect = 0;
    finalTestAccuracy = 0;
    testQuery = allData["test"][:,:QUERYLEN];
    preds = model.predict_classes(testQuery, verbose=0);
    testTargets = allData["test"][:,QUERYLEN:];

    def backToString(classes):
        return "".join([allData["oneHotMap"][c] for c in classes]);

    for i in range(0,len(preds)):
        correct = backToString([list(l).index(True) for l in list(testTargets[i])])
        guess = backToString(list(preds[i]))
        if correct == guess:
            testCorrect += 1
        if i<5:
            query = backToString([list(l).index(True) for l in list(testQuery[i])]);
            print("Q: ", query, "; Prediction: ", guess, "; Answer: ", correct, " (", correct==guess, ") ")
    finalTestAccuracy = testCorrect / len(preds);
    print("Final Test Accuracy is {}".format(finalTestAccuracy));
    resultAry = [*configInfo, finalLoss, finalAccuracy, finalValAccuracy, finalTestAccuracy];
    with open("finalResults.csv", 'a', newline='', encoding='utf-8') as csvfile:
        toWriter = csv.writer(csvfile);
        toWriter.writerow(resultAry);
    

trainingLogHeader = ["digitSize", "calcRule", "layerCount", "trainingSize", "hiddenSize", "epochSize", "modelID", "Epoch","Loss","Accuracy","Validation Accuracy"]
finalResult = ["digitSize", "calcRule", "layerCount", "trainingSize", "hiddenSize", "epochSize", "modelID", "Final Training Loss", "Final Training Accuracy", "Final Validation Accuracy", "Final Test Accuracy"]
trials = [{"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":1, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":2, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":1, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":2, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":3, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":64, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":32, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":2, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":3, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":0.8, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":0.6, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":1, "layerCount":2, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":2, "layerCount":2, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":1, "layerCount":3, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":2, "layerCount":3, "trainingSize":1, "hiddenSize":128, "modelID": 0},
        {"digitSizeID":0, "toRuleID":0, "layerCount":1, "trainingSize":1, "hiddenSize":128, "modelID": 1},
        {"digitSizeID":0, "toRuleID":1, "layerCount":3, "trainingSize":1, "hiddenSize":128, "epochSize": 500, "modelID": 0},
        {"digitSizeID":0, "toRuleID":2, "layerCount":3, "trainingSize":1, "hiddenSize":128, "epochSize": 1500,  "modelID": 0}]

with open("finalResults.csv", 'w', newline='', encoding='utf-8') as csvfile:
    toWriter = csv.writer(csvfile);
    toWriter.writerow(finalResult);

with open("trainingLog.csv", 'w', newline='', encoding='utf-8') as csvfile:
    toWriter = csv.writer(csvfile);
    toWriter.writerow(trainingLogHeader);

for t in trials:
    trainModel(**t);

