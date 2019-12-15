import numpy as np
import sklearn.metrics as metrics
import thermal_evaluation as cN
import torch



class LogMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class AccMeter(object):
    def __init__(self):
        self.sample_size = 0

    def update(self, y_true, y_pred):
        if self.sample_size==0:
            self.true = y_true
            self.pred = y_pred
            self.sample_size = self.true.shape[0]
            if self.true.ndim > 1:
                self.label_size = self.true.shape[1]
            else:
                self.label_size = 1

        else:
            self.true = np.append(self.true, y_true, axis=0)
            self.pred = np.append(self.pred, y_pred, axis=0)
            self.sample_size = self.true.shape[0]

    ''' classification error '''
    def confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.true, self.pred)
        return confusion_matrix

    def class_report(self):
        classify_report = metrics.classification_report(self.true, self.pred)
        return classify_report

    # producer's accuracy
    def overall_acc(self):
        overall_accuracy = metrics.accuracy_score(self.true, self.pred)
        return overall_accuracy

    # user's accuracy
    def each_acc(self):
        acc_for_each_class = metrics.precision_score(self.true, self.pred, average=None)
        return acc_for_each_class


    # short accuracy
    def f1_score(self):
        f1_score = metrics.accuracy_score(self.true, self.pred)
        return f1_score


    ''' regression error '''

    def mean_abs_error(self):
        mean = metrics.mean_absolute_error(self.true, self.pred, multioutput='raw_values')
        return mean

    def mean_sqr_error(self):
        mean = metrics.mean_squared_error(self.true, self.pred, multioutput='raw_values')
        return mean

    def median_abs_error(self):
        mean = metrics.median_absolute_error(self.true, self.pred)
        return mean

    def r2_score(self):
        r2_score = metrics.r2_score(self.true, self.pred, multioutput='raw_values')
        return r2_score

def field_error(true, pred):

    y_error = torch.abs(true - pred)

    L1_error = torch.mean(y_error, dim=(2, 3))
    L2_error = torch.mean(y_error ** 2, dim=(2, 3))
    Li_error = y_error.max(dim=3)[0].max(dim=2)[0]

    L_error = torch.stack((L1_error, L2_error, Li_error), dim=2).cpu().numpy()
    return L_error

