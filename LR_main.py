import csv
import numpy as np
import pandas as pd


def init_data(filename):
    dataset = []
    csv_reader = csv.reader(open(filename))
    csv_header = next(csv_reader)                                 # read header=(feature1,feature2,...,feature n,label)
    for row in csv_reader:
        dataset.append(row)                                       # read data
    dataset = [[float(x) for x in row] for row in dataset]        # change char to float
    dataset = np.array(dataset)
    feature_num = len(csv_header)                                 # obtain feature_num=n+1(n features, 1 label)
    feature = dataset[:,0:feature_num-1]                          # obtain data of n features
    label = dataset[:,-1]                                         # obtain data of labels
    return feature, label.reshape(len(label), 1)


def logistic_f(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def logistic_regression(feature, label, alpha, max_iterations, classifier):
    m, n = feature.shape                                                  # m:number of instances n：number of features
    weights = np.zeros((n,1))                                             # initialization: (1,...,1)
    for i in range(max_iterations):                                       # iterate n=max_iterations times
        A = np.dot(feature, weights)                                      # compute w^{T}x
        for j in range(len(A)):
            A[j][0] = logistic_f(A[j][0])                                 # compute h(x)=1/(1+exp^{-w^{T}x})
        E = A-label                                                       # compute E=h(x)-y
        weights = weights - alpha * np.dot(feature.T,E) / m               # update w=w-alpha*x^{T}*E
        error = np.sum(np.abs(E))
        print("Classifier[%d] iteration %d: error %f, percentage %.2f%%" % (classifier, i, error, 100*error/m))
    return weights


num_of_features=17
num_of_classes=26
# training:
# read data
data_feature, data_label = init_data("dataset/train_set.csv")
data_feature = np.insert(data_feature, 0, values=np.ones(len(data_feature)), axis=1)

# initialize w
w = np.zeros(num_of_features*num_of_classes).reshape(num_of_classes,num_of_features)

# create 26 classifiers
for i in range(num_of_classes):
    data_label_modified = []
    for j in range(len(data_label)):
            if data_label[j] == i+1:
                data_label_modified.append(1)
            else:
                data_label_modified.append(0)
    data_label_modified = np.array(data_label_modified).reshape(len(data_label_modified), 1)
    w[i] = logistic_regression(data_feature, data_label_modified, 0.085, 10000, i).reshape(1, num_of_features)
    print(w[i])

# testing:
# read data
data_feature, data_label = init_data("dataset/test_set.csv")
data_feature = np.insert(data_feature, 0, values=np.ones(len(data_feature)), axis=1)

# get prediction of label
vote_result = np.dot(data_feature, w.T)
for i in range(len(vote_result)):
    for j in range(num_of_classes):
        vote_result[i][j] = logistic_f(vote_result[i][j])
data_label_prediction = (vote_result.argmax(axis=1)+1).reshape(len(vote_result),1)

prediction = pd.DataFrame(data_label_prediction)
prediction.to_csv('result/prediction.csv')

# calculate performance: accuracy, precision, recall, and F1 score
ACC = np.sum(data_label_prediction == data_label) / len(data_label)

TP = np.zeros(num_of_classes)
FN = np.zeros(num_of_classes)
FP = np.zeros(num_of_classes)
TN = np.zeros(num_of_classes)

for i in range(num_of_classes):
    for j in range(len(data_label)):
        if data_label[j][0] == i+1 :
            if data_label_prediction[j][0] == i+1:
                TP[i] += 1
            else:
                FN[i] += 1
        else:
            if data_label_prediction[j][0] == i+1:
                FP[i] += 1
            else:
                TN[i] += 1

micro_precision = np.mean(TP) / (np.mean(TP) + np.mean(FP))
micro_recall = np.mean(TP) / (np.mean(TP) + np.mean(FN))
micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
macro_precision = np.mean(TP / (TP + FP))
macro_recall = np.mean( TP /(TP + FN))
macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

print("Performance of the classifiers on test set：")
print(" Accuracy: %.2f%%" % (100 * ACC))
print(" micro Precision: %.2f%%" % (100 * micro_precision))
print(" micro Recall: %.2f%%" % (100 * micro_recall))
print(" micro F1: %.2f%%" % (100 * micro_f1))
print(" macro Precision: %.2f%%" % (100 * macro_precision))
print(" macro Recall: %.2f%%" % (100 * macro_recall))
print(" macro F1: %.2f%%" % (100 * macro_f1))
