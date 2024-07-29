import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchmetrics



class MetricsCalculator:
    def __init__(self, num_cls, class_names, prefix_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_names = class_names
        self.num_cls = num_cls

        self.dict_of_metrics = {
            # ----- accuracy -----
            'acc_multiclass_top1_micro': torchmetrics.Accuracy(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'acc_multiclass_top5_micro': torchmetrics.Accuracy(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'acc_multiclass_top1_macro': torchmetrics.Accuracy(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'acc_multiclass_top5_macro': torchmetrics.Accuracy(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'acc_multiclass_top1_noavg': torchmetrics.Accuracy(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            'acc_multilabel_thresh0.3_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            'acc_multilabel_thresh0.5_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            'acc_multilabel_thresh0.7_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            'acc_multilabel_thresh0.3_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            'acc_multilabel_thresh0.5_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            'acc_multilabel_thresh0.7_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- precision -----
            'precision_multiclass_top1_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'precision_multiclass_top5_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'precision_multiclass_top1_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'precision_multiclass_top5_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'precision_multiclass_top1_noavg': torchmetrics.Precision(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            'precision_multilabel_thresh0.3_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            'precision_multilabel_thresh0.5_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            'precision_multilabel_thresh0.7_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            'precision_multilabel_thresh0.3_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            'precision_multilabel_thresh0.5_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            'precision_multilabel_thresh0.7_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- recall -----
            'recall_multiclass_top1_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'recall_multiclass_top5_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'recall_multiclass_top1_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'recall_multiclass_top5_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'recall_multiclass_top1_noavg': torchmetrics.Recall(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            'recall_multilabel_thresh0.3_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            'recall_multilabel_thresh0.5_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            'recall_multilabel_thresh0.7_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            'recall_multilabel_thresh0.3_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            'recall_multilabel_thresh0.5_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            'recall_multilabel_thresh0.7_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- f1 -----
            'f1_multiclass_top1_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'f1_multiclass_top5_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'f1_multiclass_top1_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'f1_multiclass_top5_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'f1_multiclass_top1_noavg': torchmetrics.F1Score(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            'f1_multilabel_thresh0.3_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            'f1_multilabel_thresh0.5_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            'f1_multilabel_thresh0.7_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            'f1_multilabel_thresh0.3_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            'f1_multilabel_thresh0.5_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            'f1_multilabel_thresh0.7_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- auroc score -----
            # 'auroc_score_multiclass_noavg': torchmetrics.AUROC(num_classes=num_cls, average=None, task='multiclass').to(self.device), # 'summarizes the ROC curve into a single number that describes the performance of a model for multiple thresholds at the same time'
            'auroc_score_multiclass_macro': torchmetrics.AUROC(num_classes=num_cls, average='macro', task='multiclass').to(self.device),
            # 'auroc_score_multilabel_thresh0.3_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            # 'auroc_score_multilabel_thresh0.5_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            # 'auroc_score_multilabel_thresh0.7_micro': torchmetrics.AUROC(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            # 'auroc_score_multilabel_thresh0.3_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            # 'auroc_score_multilabel_thresh0.5_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            # 'auroc_score_multilabel_thresh0.7_macro': torchmetrics.AUROC(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
    
            # ----- ap score -----
            # 'ap_score_multiclass_noavg': torchmetrics.AveragePrecision(num_classes=num_cls, average=None, task='multiclass').to(self.device), # summarizes the precision-recall curve as an weighted mean of precisions at each threshold, with the difference in recall from the previous threshold as weight
            'ap_score_multiclass_macro': torchmetrics.AveragePrecision(num_classes=num_cls, average='macro', task='multiclass').to(self.device),
            # 'ap_score_multilabel_thresh0.3_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3),
            # 'ap_score_multilabel_thresh0.5_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5),
            # 'ap_score_multilabel_thresh0.7_micro': torchmetrics.AveragePrecision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7),
            # 'ap_score_multilabel_thresh0.3_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3),
            # 'ap_score_multilabel_thresh0.5_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5),
            # 'ap_score_multilabel_thresh0.7_macro': torchmetrics.AveragePrecision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7),
        }

        self.dict_of_plottings = {
            # ----- plotting -----
            'confusion_matrix_multiclass': torchmetrics.ConfusionMatrix(num_classes=num_cls, task='multiclass', normalize=None).to(self.device),
            'roc_curve_multiclass': torchmetrics.ROC(num_classes=num_cls, task='multiclass').to(self.device),
            'pr_curve_multiclass': torchmetrics.PrecisionRecallCurve(num_classes=num_cls, task='multiclass').to(self.device)
        }

        self.prefix_name = prefix_name

    def generate_dict_of_metrics(self, logits, y):
        ret_dict = {}

        for metric_name, metric_op in self.dict_of_metrics.items():
            # print('[**] metric_name: ', metric_name)

            metric_name = ''.join([self.prefix_name, '_', metric_name])
            if logits is None:
                ret_dict[metric_name] = -1
            else:
                if 'multilabel' in metric_name:
                    metric_val = metric_op(logits, F.one_hot(y, num_classes=self.num_cls).float())
                else:
                    metric_val = metric_op(logits, y)
                ret_dict[metric_name] = metric_val.type(torch.FloatTensor)

        return ret_dict

    def generate_plots(self, logits, y, save_dir=None):
        self.dict_of_plottings['confusion_matrix_multiclass'].update(logits, y)
        # cm = self.dict_of_plottings['confusion_matrix_multiclass'](logits, y)
        fig_confusion_matrix_multiclass, ax_confusion_matrix_multiclass = self.dict_of_plottings['confusion_matrix_multiclass'].plot(add_text=True, labels=self.class_names) # `labels` is a list of strings, `cmap` is matplotlib colormap to use for the confusion matrix 

        # ax_confusion_matrix_multiclass.set_title('Confusion matrix')
        # ax_confusion_matrix_multiclass.set_xlabel('Predicted')
        # ax_confusion_matrix_multiclass.set_ylabel('Groundtruth')
        # ax_confusion_matrix_multiclass.set_ylim(-5, 100)
        # ax_confusion_matrix_multiclass.set_xscale("linear")
        # ax_confusion_matrix_multiclass.set_yscale("linear")

        # Create color bar
        # cbar = ax_confusion_matrix_multiclass.figure.colorbar(cm, ax=ax_confusion_matrix_multiclass)
        # cbar.ax.set_ylabel(cbarlabel="", rotation=-90, va="bottom")
        # Let the horizontal axes labeling appear on top
        ax_confusion_matrix_multiclass.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment
        ax_confusion_matrix_multiclass.set_xticklabels(ax_confusion_matrix_multiclass.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        # Turn spines off and create white grid
        ax_confusion_matrix_multiclass.spines[:].set_visible(False)
        ax_confusion_matrix_multiclass.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax_confusion_matrix_multiclass.tick_params(which="minor", bottom=False, left=False)

        fig_confusion_matrix_multiclass.set_size_inches(42, 42)
        fig_confusion_matrix_multiclass.tight_layout()


        self.dict_of_plottings['roc_curve_multiclass'].update(logits, y)
        # fpr, tpr, thresholds = self.dict_of_plottings['roc_curve_multiclass'](logits, y)
        fig_roc_curve_multiclass, ax_roc_curve_multiclass = self.dict_of_plottings['roc_curve_multiclass'].plot(score=True)

        self.dict_of_plottings['pr_curve_multiclass'].update(logits, y)
        # precision, recall, thresholds = self.dict_of_plottings['pr_curve_multiclass'](logits, y)
        fig_pr_curve_multiclass, ax_pr_curve_multiclass = self.dict_of_plottings['pr_curve_multiclass'].plot(score=True)

        fig_confusion_matrix_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'cm.png'])))
        fig_roc_curve_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'roc_curve.png'])))
        fig_pr_curve_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'pr_curve.png'])))

