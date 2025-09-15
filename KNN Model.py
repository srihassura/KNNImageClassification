# -*- coding: utf-8 -*-
"""
NCSSM CS4320 Final Project
@author: srihas surapaneni
"""

#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#functions
#distance
def distance(A, b):
    distances = (np.sqrt(np.sum((A - b)**2, axis = 1)))
    return distances

#A
def createArray(df, dropCol):
    tdf = df.drop(dropCol, axis = 1)
    A = tdf.to_numpy()
    return A

#count neighbors
def countNeighbors(k, sortedIndices, Alabels):
    kNearestLabels = [Alabels[i] for i in sortedIndices[:k]]

    unique, counts = np.unique(kNearestLabels, return_counts=True)

    return unique[np.argmax(counts)]

#prediction
def prediction(k, A, b, Alabels):
    predictions = []
    for i in range(len(b)):
        print(i)
        testPoint = b[i]

        distances = distance(A, testPoint)

        sortedIndices = np.argsort(distances)

        prediction = countNeighbors(k, sortedIndices, Alabels)
        predictions.append(prediction)
    return predictions

#mean and sd
def meanSD(df, dropCol):
    mean = []
    sd = []
    tdf = df.drop(dropCol, axis = 1)
    for col in tdf.columns:
        mean.append(tdf[col].mean())
        sd.append(tdf[col].std())
    return mean, sd

#standardize
def standardize(df, mean, sd, dropCol):
    tdf = df.drop(dropCol, axis = 1)
    for col in tdf.columns:
        i = tdf.columns.get_loc(col)
        tdf[col] = (tdf[col] - mean[i]) / sd[i]
    tdf[dropCol] = df[dropCol]

    return tdf

#confusion matrix
def confusionMatrix(predictions, blabels):
    classes = sorted(list(set(blabels)))
    numClasses = len(classes)
    confusionMatrix = np.zeros((numClasses, numClasses), dtype=int)
    metrics = {}

    for i in range(len(blabels)):
        trueidx = classes.index(blabels[i])
        predidx = classes.index(predictions[i])
        confusionMatrix[trueidx][predidx] += 1
    
    for i in range(numClasses):
        TP = confusionMatrix[i][i]
        FP = np.sum(confusionMatrix[:, i]) - TP
        FN = np.sum(confusionMatrix[i, :]) - TP
        TN = np.sum(confusionMatrix) - (TP + FP + FN)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * ((precision * recall) / (precision + recall))

        metrics[classes[i]] = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        }
    
    misclassifications = np.sum(confusionMatrix) - np.trace(confusionMatrix)
    overallAccuracy = np.trace(confusionMatrix) / np.sum(confusionMatrix)
    return metrics, overallAccuracy, misclassifications

#macro-average metrics
def macroAverage(metrics):
    precisionSum = 0
    recallSum = 0
    f1Sum = 0
    accuracySum = 0

    for i in metrics:
        precisionSum += metrics[i]['Precision']
        recallSum += metrics[i]['Recall']
        f1Sum += metrics[i]['F1']
        accuracySum += metrics[i]['Accuracy']

    numClasses = len(metrics)

    return{
        'Macro-Precision': precisionSum / numClasses,
        'Macro-Recall': recallSum / numClasses,
        'Macro-F1': f1Sum / numClasses,
        'Macro-Accuracy': accuracySum / numClasses
    }

#plot confusion matrix
def plotConfusionMatrix(predictions, blabels):
    classes = sorted(list(set(blabels)))
    numClasses = len(classes)
    confMatrix = np.zeros((numClasses, numClasses), dtype=int)
    
    for i in range(len(blabels)):
        trueidx = classes.index(blabels[i])
        predidx = classes.index(predictions[i])
        confMatrix[trueidx][predidx] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(numClasses)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = confMatrix.max() / 2
    for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
            plt.text(j, i, confMatrix[i, j],
                     horizontalalignment="center",
                     color="white" if confMatrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#main
#data input
traindf = pd.read_csv(str(input("Enter Training File: ")))
validdf = pd.read_csv(str(input("Enter Validation File: ")))
dropCol = "label"

#parameters
k = 15
m = len(traindf)
testm = len(validdf)
n = len(traindf.columns) - 1
testn = len(validdf.columns) - 1
print(f"m = {m}, n = {n}, testm = {testm}, testn = {testn}")

#A and b
A = createArray(traindf, dropCol)
b = createArray(validdf, dropCol)
Alabels = traindf[dropCol].values
blabels = validdf[dropCol].values

#prediction
predictions = prediction(k, A, b, Alabels)
results = validdf.copy()
results["Predicted"] = predictions
print("\First 10 Predictions:")
sample_results = results[[dropCol, 'Predicted']].head(10)
print(sample_results)

#confusion matrix
metrics, overallAccuracy, misclassifications = confusionMatrix(predictions, blabels)

print("\nConfusion Matrix Metrics:")
for cls in metrics:
    print(f"\nClass {cls} Metrics:")
    print(f"  True Positives (TP): {metrics[cls]['TP']}")
    print(f"  True Negatives (TN): {metrics[cls]['TN']}")
    print(f"  False Positives (FP): {metrics[cls]['FP']}")
    print(f"  False Negatives (FN): {metrics[cls]['FN']}")
    print(f"  Precision: {metrics[cls]['Precision']:.4f}")
    print(f"  Recall: {metrics[cls]['Recall']:.4f}")
    print(f"  F1 Score: {metrics[cls]['F1']:.4f}")
    print(f"  Accuracy: {metrics[cls]['Accuracy']:.4f}")

print(f"\nOverall Accuracy: {overallAccuracy:.4f}")
print(f"Total Misclassifications: {misclassifications}")

#macro-average metrics
macroMetrics = macroAverage(metrics)
print("\nMacro-average Metrics:")
for metric, value in macroMetrics.items():
    print(f"  {metric}: {value:.4f}")

#plot
plotConfusionMatrix(predictions, blabels)