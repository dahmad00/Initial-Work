import tensorflow as tf
import numpy as np
import pandas as pd

# Find accuracy of model
def find_accuracy(test,pred):
    correct = 0
    total = len(test)

    for i in range(len(test)):
        if test[i] == pred[i]:
            correct += 1

    return correct/total 


# Map ANN outputs to classes
def get_labels(y_pred_ann):
    labels = []

    for pred in y_pred_ann:
        max_index = 0

        for i in range(len(pred)):
            if pred[i] > pred[max_index]:
                max_index = i
        
        labels.append(max_index)

    return labels

# This callback prints accuracy by epoch information after each epoch
class Save_Accuracy_By_Epoch(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.X_Test = test_data[0]
        self.Y_Test = test_data[1]
        self.accuracies = []
        self.epochs = []
        
    def on_epoch_end(self, epoch, logs = None):
        y_pred = self.model.predict(self.X_Test)

        if epoch == 4:
            pass

        y_pred = get_labels(y_pred)
        accuracy = find_accuracy(self.Y_Test, y_pred) 
        self.epochs.append(epoch+1)
        self.accuracies.append(accuracy)

        

        print(self.epochs)
        print(self.accuracies)

        
# This callback prints metrics for every class after each epoch
class Save_Multiclass_Metrics_By_Epoch(tf.keras.callbacks.Callback):
    def __init__(self, test_data, n_classes, save_after = 10, save_csv_path = 'results.csv'):
        self.X_Test = test_data[0]
        self.Y_Test = test_data[1]
        self.epochs = []
        self.n_classes = n_classes
        self.save_after = save_after
        self.save_csv_path = save_csv_path

        self.mat_sensitivity = []
        self.mat_specificity = []
        self.mat_precision = []
        self.mat_recall = []
        self.mat_accuracy = []
        self.mat_f1 = []

        for i in range(n_classes):
            self.mat_sensitivity.append([])
            self.mat_specificity.append([])
            self.mat_precision.append([])
            self.mat_recall.append([])
            self.mat_accuracy.append([])
            self.mat_f1.append([])

        
    def on_epoch_end(self, epoch, logs = None):
        y_pred = self.model.predict(self.X_Test)
        y_pred = get_labels(y_pred)


        for i in range(self.n_classes):
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for j in range(len(y_pred)):
                if self.Y_Test[j] == i and y_pred[j] == i:
                    TP += 1
                elif self.Y_Test[j] != i and y_pred[j] == i:
                    FP += 1
                elif self.Y_Test[j] == i and y_pred[j] != i:
                    FN += 1
                elif self.Y_Test[j] != i and y_pred[j] != i:
                    TN += 1


            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else -1
            specificity = TN / (TN + FP) if (TN + FP) > 0 else -1
            precision = TP / (TP + FP) if (TP + FP) > 0 else -1
            recall = TP / (TP + FN) if (TP + FN) > 0 else -1
            accuracy = (TP + TN) / (TP + FN + TN + FP) 
            f1 = (precision * recall) / (precision + recall) if (precision + recall) > 0 else -1

            self.mat_sensitivity[i].append(sensitivity)
            self.mat_specificity[i].append(specificity)
            self.mat_precision[i].append(precision)
            self.mat_recall[i].append(recall)
            self.mat_accuracy[i].append(accuracy)
            self.mat_f1[i].append(f1)
        
        self.epochs.append(epoch+1)

        if (epoch + 1) % self.save_after == 0:
            save_to_csv_file(self.save_csv_path, self.mat_sensitivity, self.mat_specificity, self.mat_precision, self.mat_recall, self.mat_accuracy, self.mat_f1)
            pass

def save_to_csv_file(path, mat_sensitivity, mat_specificity, mat_precision, mat_recall, mat_accuracy, mat_f1):
    mat_sensitivity = np.transpose(mat_sensitivity)
    mat_specificity = np.transpose(mat_specificity)
    mat_precision = np.transpose(mat_precision)
    mat_recall = np.transpose(mat_recall)
    mat_accuracy = np.transpose(mat_accuracy)
    mat_f1 = np.transpose(mat_f1)

    mat_join = np.concatenate((mat_sensitivity, mat_specificity, mat_precision, mat_recall, mat_accuracy, mat_f1), axis = 1)

    n = mat_sensitivity.shape[1]


    col_array_sensitivity = list(range(n))
    col_array_specificity = list(range(n))
    col_array_precision = list(range(n))
    col_array_recall = list(range(n))
    col_array_accuracy = list(range(n))
    col_array_f1 = list(range(n))
    
    
    for i in range(n):
        col_array_sensitivity[i] = 'sensitivity Class ' + str(col_array_sensitivity[i])
        col_array_specificity[i] = 'specificity Class ' + str(col_array_specificity[i])
        col_array_precision[i] = 'precision Class ' + str(col_array_precision[i])
        col_array_recall[i] = 'recall Class ' + str(col_array_recall[i])
        col_array_accuracy[i] = 'accuracy Class ' + str(col_array_accuracy[i])
        col_array_f1[i] = 'f1 Class ' + str(col_array_f1[i])

    cols = col_array_sensitivity + col_array_specificity + col_array_precision + col_array_recall + col_array_accuracy + col_array_f1



    df = pd.DataFrame(
        columns = cols,
        data  = mat_join
    )

    df.to_csv(path, index= False)
    













            
        