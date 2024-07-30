import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
matplotlib.use('Agg')



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
            # 'acc_multilabel_thresh0.3_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            # 'acc_multilabel_thresh0.5_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            # 'acc_multilabel_thresh0.7_micro': torchmetrics.Accuracy(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            # 'acc_multilabel_thresh0.3_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            # 'acc_multilabel_thresh0.5_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            # 'acc_multilabel_thresh0.7_macro': torchmetrics.Accuracy(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- precision -----
            'precision_multiclass_top1_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'precision_multiclass_top5_micro': torchmetrics.Precision(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'precision_multiclass_top1_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'precision_multiclass_top5_macro': torchmetrics.Precision(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'precision_multiclass_top1_noavg': torchmetrics.Precision(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            # 'precision_multilabel_thresh0.3_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            # 'precision_multilabel_thresh0.5_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            # 'precision_multilabel_thresh0.7_micro': torchmetrics.Precision(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            # 'precision_multilabel_thresh0.3_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            # 'precision_multilabel_thresh0.5_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            # 'precision_multilabel_thresh0.7_macro': torchmetrics.Precision(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- recall -----
            'recall_multiclass_top1_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'recall_multiclass_top5_micro': torchmetrics.Recall(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'recall_multiclass_top1_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'recall_multiclass_top5_macro': torchmetrics.Recall(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'recall_multiclass_top1_noavg': torchmetrics.Recall(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            # 'recall_multilabel_thresh0.3_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            # 'recall_multilabel_thresh0.5_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            # 'recall_multilabel_thresh0.7_micro': torchmetrics.Recall(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            # 'recall_multilabel_thresh0.3_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            # 'recall_multilabel_thresh0.5_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            # 'recall_multilabel_thresh0.7_macro': torchmetrics.Recall(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
            # ----- f1 -----
            'f1_multiclass_top1_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=1).to(self.device),
            'f1_multiclass_top5_micro': torchmetrics.F1Score(num_classes=num_cls, average='micro', task='multiclass', top_k=5).to(self.device),
            'f1_multiclass_top1_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=1).to(self.device),
            'f1_multiclass_top5_macro': torchmetrics.F1Score(num_classes=num_cls, average='macro', task='multiclass', top_k=5).to(self.device),
            # 'f1_multiclass_top1_noavg': torchmetrics.F1Score(num_classes=num_cls, average=None, task='multiclass').to(self.device),
            # 'f1_multilabel_thresh0.3_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.3).to(self.device),
            # 'f1_multilabel_thresh0.5_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.5).to(self.device),
            # 'f1_multilabel_thresh0.7_micro': torchmetrics.F1Score(num_labels=num_cls, average='micro', task='multilabel', threshold=0.7).to(self.device),
            # 'f1_multilabel_thresh0.3_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.3).to(self.device),
            # 'f1_multilabel_thresh0.5_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.5).to(self.device),
            # 'f1_multilabel_thresh0.7_macro': torchmetrics.F1Score(num_labels=num_cls, average='macro', task='multilabel', threshold=0.7).to(self.device),
    
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

    def generate_plots(self, logits, y, save_dir=None, suffix=''):
        self.dict_of_plottings['confusion_matrix_multiclass'].update(logits, y)
        # cm = self.dict_of_plottings['confusion_matrix_multiclass'](logits, y)
        # print('[**] {} - Sum of cm: {}'.format(self.prefix_name, torch.sum(cm)))
        fig_confusion_matrix_multiclass, ax_confusion_matrix_multiclass = self.dict_of_plottings['confusion_matrix_multiclass'].plot(add_text=True, labels=self.class_names) # `labels` is a list of strings, `cmap` is matplotlib colormap to use for the confusion matrix 

        # ax_confusion_matrix_multiclass.set_title('Confusion matrix')
        # ax_confusion_matrix_multiclass.set_xlabel('Predicted')
        # ax_confusion_matrix_multiclass.set_ylabel('Groundtruth')
        # ax_confusion_matrix_multiclass.set_ylim(-5, 100)
        # ax_confusion_matrix_multiclass.set_xscale("linear")
        # ax_confusion_matrix_multiclass.set_yscale("linear")

        # Create color bar
        axes_image = ax_confusion_matrix_multiclass.get_images()[0] # mappable object: AxesImage of Axes - get an image inside an axis
        axes_image.set(cmap='BuPu')
        # ax_confusion_matrix_multiclass.figure.colorbar(axes_image)
        cbar = fig_confusion_matrix_multiclass.figure.colorbar(axes_image, ax=ax_confusion_matrix_multiclass, drawedges=False, fraction=0.02, aspect=50)
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")
        # Let the horizontal axes labeling appear on top
        ax_confusion_matrix_multiclass.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment
        ax_confusion_matrix_multiclass.set_xticklabels(ax_confusion_matrix_multiclass.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        # Turn spines off and create white grid
        ax_confusion_matrix_multiclass.spines[:].set_visible(False)
        ax_confusion_matrix_multiclass.set_xticks(np.arange(len(self.class_names) + 1)-.5, minor=True)
        ax_confusion_matrix_multiclass.set_yticks(np.arange(len(self.class_names) + 1)-.5, minor=True)
        ax_confusion_matrix_multiclass.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax_confusion_matrix_multiclass.tick_params(which="minor", bottom=False, left=False)

        ax_confusion_matrix_multiclass.tick_params(axis='both', which='major', labelsize=20)
        ax_confusion_matrix_multiclass.tick_params(axis='both', which='minor', labelsize=18)

        texts = annotate_heatmap(axes_image, valfmt="{x:.0f}")

        fig_confusion_matrix_multiclass.set_size_inches(32, 30)
        # fig_confusion_matrix_multiclass.tight_layout()


        self.dict_of_plottings['roc_curve_multiclass'].update(logits, y)
        # fpr, tpr, thresholds = self.dict_of_plottings['roc_curve_multiclass'](logits, y)
        fig_roc_curve_multiclass, ax_roc_curve_multiclass = self.dict_of_plottings['roc_curve_multiclass'].plot(score=True)

        ax_roc_curve_multiclass.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig_roc_curve_multiclass.set_size_inches(25, 20)

        self.dict_of_plottings['pr_curve_multiclass'].update(logits, y)
        # precision, recall, thresholds = self.dict_of_plottings['pr_curve_multiclass'](logits, y)
        fig_pr_curve_multiclass, ax_pr_curve_multiclass = self.dict_of_plottings['pr_curve_multiclass'].plot(score=True)

        ax_pr_curve_multiclass.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig_pr_curve_multiclass.set_size_inches(25, 20)

        fig_confusion_matrix_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'cm_', suffix, '.png'])), dpi=600, bbox_inches="tight")
        fig_roc_curve_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'roc_curve_', suffix, '.png'])), dpi=600, bbox_inches="tight")
        fig_pr_curve_multiclass.savefig(os.path.join(save_dir, ''.join([self.prefix_name, '_', 'pr_curve_', suffix, '.png'])), dpi=600, bbox_inches="tight")

        plt.close(fig_confusion_matrix_multiclass)
        plt.close(fig_roc_curve_multiclass)
        plt.close(fig_pr_curve_multiclass)

        # Reset global states of `torchmetrics.Metric`. Ref: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
        self.dict_of_plottings['confusion_matrix_multiclass'].reset()
        self.dict_of_plottings['roc_curve_multiclass'].reset()
        self.dict_of_plottings['pr_curve_multiclass'].reset()

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap. Reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=18.0)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # TODO: remove previous plotted texts by torchmetrics. Reference: https://stackoverflow.com/questions/26005138/how-can-you-clear-a-matplotlib-text-box-that-was-previously-drawn
    for txt in im.axes.texts:
        txt.set_visible(False)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw) # keep references for modifying
            texts.append(text)

    return texts