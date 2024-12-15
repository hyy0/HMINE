import numpy as np

def recall(true, pred):
    TP = np.sum((true == 1) & (pred == 1))
    FN = np.sum((true == 1) & (pred == 0))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def precision(true, pred):
    TP = np.sum((true == 1) & (pred == 1))
    FP = np.sum((true == 0) & (pred == 1))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def false_positive_rate(true, pred):
    FP = np.sum((true == 0) & (pred == 1))
    TN = np.sum((true == 0) & (pred == 0))
    return FP / (FP + TN) if (FP + TN) > 0 else 0

def F1_score(true, pred):
    TP = np.sum((true == 1) & (pred == 1))
    FP = np.sum((true == 0) & (pred == 1))
    FN = np.sum((true == 1) & (pred == 0))
    
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0