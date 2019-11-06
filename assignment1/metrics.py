def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i] == True:
            tp += 1
        elif prediction[i] == ground_truth[i] == False:
            tn += 1
        elif prediction[i] == True and ground_truth[i] == False:
            fp += 1
        elif prediction[i] == False and ground_truth[i] == True:
            fn +=1

    accuracy = (tp + tn) / len(prediction)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    positive = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            positive += 1

    return positive / len(prediction)