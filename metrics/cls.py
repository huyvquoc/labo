import torchmetrics


class MetricsCalculator:
    def __init__(self, num_cls, class_names, prefix_name):

        self.class_names = class_names

        self.dict_of_metrics = {
            # ----- accuracy -----
            'acc_multiclass_top1_micro': torchmetrics.Accuracy(num_classes=num_cls, average='micro', task='multiclass', top_k=1),
            'acc_multiclass_top5_micro': torchmetrics.Accuracy(num_classes=num_cls, average='micro', task='multiclass', top_k=5),
            'acc_multiclass_top1_macro': torchmetrics.Accuracy(num_classes=num_cls, average='macro', task='multiclass', top_k=1),
            'acc_multiclass_top5_macro': torchmetrics.Accuracy(num_classes=num_cls, average='macro', task='multiclass', top_k=5),
            'acc_multiclass_top1_noavg': torchmetrics.Accuracy(num_classes=num_cls, average=None, task='multiclass'),
            'acc_multilabel_thresh0.3_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'acc_multilabel_thresh0.5_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'acc_multilabel_thresh0.7_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'acc_multilabel_thresh0.3_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'acc_multilabel_thresh0.5_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'acc_multilabel_thresh0.7_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- precision -----
            'precision_multiclass_top1_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=1),
            'precision_multiclass_top5_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=5),
            'precision_multiclass_top1_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=1),
            'precision_multiclass_top5_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=5),
            'precision_multiclass_top1_noavg': torchmetrics.Precision(num_classes=num_cls, average=None, task='multiclass'),
            'precision_multilabel_thresh0.3_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'precision_multilabel_thresh0.5_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'precision_multilabel_thresh0.7_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'precision_multilabel_thresh0.3_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'precision_multilabel_thresh0.5_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'precision_multilabel_thresh0.7_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- recall -----
            'recall_multiclass_top1_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=1),
            'recall_multiclass_top5_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=5),
            'recall_multiclass_top1_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=1),
            'recall_multiclass_top5_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=5),
            'recall_multiclass_top1_noavg': torchmetrics.Recall(num_classes=num_cls, average=None, task='multiclass'),
            'recall_multilabel_thresh0.3_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'recall_multilabel_thresh0.5_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'recall_multilabel_thresh0.7_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'recall_multilabel_thresh0.3_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'recall_multilabel_thresh0.5_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'recall_multilabel_thresh0.7_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- f1 -----
            'f1_multiclass_top1_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=1),
            'f1_multiclass_top5_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=5),
            'f1_multiclass_top1_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=1),
            'f1_multiclass_top5_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=5),
            'f1_multiclass_top1_noavg': torchmetrics.F1Score(num_classes=num_cls, average=None, task='multiclass'),
            'f1_multilabel_thresh0.3_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'f1_multilabel_thresh0.5_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'f1_multilabel_thresh0.7_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'f1_multilabel_thresh0.3_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'f1_multilabel_thresh0.5_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'f1_multilabel_thresh0.7_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- auroc score -----
            'auroc_score_multiclass_micro': torchmetrics.AUROC(num_classes=num_cls, average='micro', task='multiclass'), # 'summarizes the ROC curve into a single number that describes the performance of a model for multiple thresholds at the same time'
            'auroc_score_multiclass_macro': torchmetrics.AUROC(num_classes=num_cls, average='macro', task='multiclass'),
            'auroc_score_multilabel_thresh0.3_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'auroc_score_multilabel_thresh0.5_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'auroc_score_multilabel_thresh0.7_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'auroc_score_multilabel_thresh0.3_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'auroc_score_multilabel_thresh0.5_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'auroc_score_multilabel_thresh0.7_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- ap score -----
            'ap_score_multiclass_micro': torchmetrics.AveragePrecision(num_classes=num_cls, average='micro', task='multiclass'), # summarizes the precision-recall curve as an weighted mean of precisions at each threshold, with the difference in recall from the previous threshold as weight
            'ap_score_multiclass_macro': torchmetrics.AveragePrecision(num_classes=num_cls, average='macro', task='multiclass'),
            'ap_score_multilabel_thresh0.3_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            'ap_score_multilabel_thresh0.5_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            'ap_score_multilabel_thresh0.7_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            'ap_score_multilabel_thresh0.3_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            'ap_score_multilabel_thresh0.5_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            'ap_score_multilabel_thresh0.7_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
        }

        self.dict_of_plottings = {
            # ----- plotting -----
            'confusion_matrix_multiclass': torchmetrics.ConfusionMatrix(num_classes=num_cls, task='multiclass', normalize=None),
            'roc_curve_multiclass': torchmetrics.ROC(num_classes=num_cls, task='multiclass'),
            'pr_curve_multiclass': torchmetrics.PrecisionRecallCurve(num_classes=num_cls, task='multiclass')
        }

        self.prefix_name = prefix_name

    def generate_dict_of_metrics(self, logits, y):
        ret_dict = {}

        for metric_name, metric_op in self.dict_of_metrics.items():
            metric_name = ''.join(self.prefix_name, '_', metric_name)
            if logits is None:
                ret_dict[metric_name] = -1
            else:
                metric_val = metric_op(logits, y)
                ret_dict[metric_name] = metric_val

        return ret_dict

    def generate_plots(self, logits, y):
        self.dict_of_plottings['confusion_matrix_multiclass'].update(logits, y)
        # cm = self.dict_of_plottings['confusion_matrix_multiclass'](logits, y)
        fig_confusion_matrix_multiclass, ax_confusion_matrix_multiclass = self.dict_of_plottings['confusion_matrix_multiclass'].plot(add_text=True, labels=self.class_names) # `labels` is a list of strings, `cmap` is matplotlib colormap to use for the confusion matrix 

        self.dict_of_plottings['roc_curve_multiclass'].update(logits, y)
        # fpr, tpr, thresholds = self.dict_of_plottings['roc_curve_multiclass'](logits, y)
        fig_roc_curve_multiclass, ax_roc_curve_multiclass = self.dict_of_plottings['roc_curve_multiclass'].plot(score=True)

        self.dict_of_plottings['pr_curve_multiclass'].update(logits, y)
        # precision, recall, thresholds = self.dict_of_plottings['pr_curve_multiclass'](logits, y)
        fig_pr_curve_multiclass, ax_pr_curve_multiclass = self.dict_of_plottings['pr_curve_multiclass'].plot(score=True)

