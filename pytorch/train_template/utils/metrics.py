import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.threshold = 0.6
        self.IoU = []
        self.FN_TP = []

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Accuracy_per_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc

    # ------------------mAP 추가------------------
    def _get_TP_FP(self):
        IoU = np.array(self.IoU)
        self.FN_TP = np.array(self.FN_TP)

        threshold = self.threshold
        TP = np.zeros_like(IoU)
        FP = np.zeros_like(IoU)

        TP[IoU >= threshold] = 1
        TP[np.isnan(IoU)] = 0

        FP[IoU >= threshold] = 0
        FP[np.isnan(IoU)] = 0
        FP[(IoU < threshold) & (self.FN_TP == 0)] = 1

        return TP, FP

    def Mean_Average_Precision(self):
        TP, FP = self._get_TP_FP()
        FN_TP = self.FN_TP
        # cumsum은 누적합을 의미
        TP_cumsum = np.cumsum(TP, axis=0)
        FP_cumsum = np.cumsum(FP, axis=0)
        # FN_TP_cumsum = np.cumsum(FN_TP, axis=0)
        FN_TP_sum = np.sum(FN_TP, axis=0)

        TP_cumsum = TP_cumsum.T
        FP_cumsum = FP_cumsum.T
        FN_TP_sum = FN_TP_sum.T
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        recalls = []

        for i in range(self.num_class):
            recall = TP_cumsum[i] / FN_TP_sum[i]
            recalls.append(recall)
        recalls = np.array(recalls)

        # print("TP", TP.T[1])
        # print("TP_cumsum",TP_cumsum[1])
        # print("FP",FP_cumsum[1])
        # print("FN_TP",FN_TP_sum[1])
        # print("precisions",precisions[1])
        # print("recalls",recalls[1])

        average_precisions = []

        for i in range(len(precisions)):
            precision = precisions[i]
            recall = recalls[i]
            remove_idx = []
            for j in range(len(precision)):
                if np.isnan(precision[j]):
                    remove_idx.append(j)

            precision = np.delete(precision, remove_idx)
            recall = np.delete(recall, remove_idx)

            # s = recall.argsort()
            # recall = recall[s]
            # precision = precision[s]

            ap = np.trapz(precision, recall)
            average_precisions.append(ap)

        remove_idx = []
        for i in range(len(average_precisions)):
            if np.isnan(average_precisions[i]):
                remove_idx.append(i)

        average_precisions = np.delete(average_precisions, remove_idx)
        average_precisions = np.abs(average_precisions)
        mAP = sum(average_precisions) / len(average_precisions)
        return mAP

    # ------------------mAP 추가------------------

    def Mean_dice_coef(self, class_labels):
        # mean f1 score
        dice = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0))
        # dice2 = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))

        all_dice = 2*np.nanmean(dice)
        each_class_dice = dict([(value, key) for key, value in class_labels.items()])
        for idx in range(0, len(class_labels)):  # Except Background (Except 0)
            each_class_dice[class_labels[idx]] = 2 * np.nanmean(dice[idx])

        return all_dice, each_class_dice

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        MIoU = np.nanmean(MIoU)
        return MIoU

    def Mean_Intersection_over_Union_per_Class(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _Intersection_over_Union(self, confusion_matrix):
        IoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        return IoU

    def _FN_TP(self, confusion_matrix):
        _FN_TP = np.sum(confusion_matrix, axis=0)
        _FN_TP[_FN_TP > 0] = 1
        return _FN_TP

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        batch_confusion_matrix = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += batch_confusion_matrix
        batch_IoU = self._Intersection_over_Union(batch_confusion_matrix)
        self.IoU.append(batch_IoU)
        batch_FN_TP = self._FN_TP(batch_confusion_matrix)
        self.FN_TP.append(batch_FN_TP)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.IoU = []
        self.FN_TP = []




