#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, OneHotEncoder

def print_performance(labels, predictions):
   print("Accuracy = {}".format(accuracy_score(labels, predictions)))
   print("Precision = {}".format(precision_score(labels, predictions)))
   print("Recall = {}".format(recall_score(labels, predictions)))
   (tp, fp), (fn, tn)  = confusion_matrix(labels, predictions)
   print("Confusion matrix: tp {}, fp {}, fn {}, tn {}".format(tp, fp, fn, tn))

def pd_df_multi_class_confusion_matrix(pd_s_target, pd_s_predict):
   class_sample_ = pd_s_target.unique()
   cm = confusion_matrix(pd_s_target, pd_s_predict, labels=class_sample_)
   multi_columns = zip(['Predicted label']*(len(class_sample_)), class_sample_)
   multi_index = zip(['Actual label']*(len(class_sample_)), class_sample_)
   multi_columns = pd.MultiIndex.from_tuples(list(multi_columns))
   multi_index = pd.MultiIndex.from_tuples(list(multi_index))
   return pd.DataFrame(cm, columns=multi_columns, index=multi_index)

# Compute ROC curve and ROC area for each class
def roc_curve_multiclass(pd_s_target, pd_s_predict):
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   class_samples_ = pd_s_target.unique()
   # Binarize the output
   np_target = label_binarize(pd_s_target, classes=class_samples_)
   np_predict = label_binarize(pd_s_predict, classes=class_samples_)

   for sample, unique in zip(class_samples_, range(len(class_samples_))):
      fpr[sample], tpr[sample], _ = roc_curve(np_target[:, unique], np_predict[:, unique])
      roc_auc[sample] = auc(fpr[sample], tpr[sample])
   return fpr, tpr, roc_auc

def plot_auc_roc_multiclass(fpr, tpr, roc_auc, class_samples_):
   plt.figure()
   lw = len(class_samples_)
   for i in class_samples_:
      plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
      )

   plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel("False Positive Rate")
   plt.ylabel("True Positive Rate")
   plt.title("Receiver operating characteristic example")
   plt.legend(loc="lower right")
   plt.show()

