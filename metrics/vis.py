import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def plot_concept_attn_score_single_sample(gt_concepts_raw, pred_concepts_raw, plotting_dir, sample_id, abs_img_path):
    '''
    Args:
    - pred_concepts_raw: concept sentences have been sorted in an decreasing order of concept score
    '''
    # TODO: top 10 predicted concept scores, which one is correct, which one is incorrect
    # Example sentences with corresponding colors (green for positive, red for negative)
    top10_pred_concepts = pred_concepts_raw[:10]

    sentences = []
    for pred_concept in top10_pred_concepts:
        if pred_concept in gt_concepts_raw:
            sent_tup = (pred_concept, "green")
        else:
            sent_tup = (pred_concept, "red")
        sentences.append(sent_tup)

    # sentences = [
    #     ("This is a good sentence.", "green"),
    #     ("This is a bad sentence.", "red"),
    #     ("Everything is fine here.", "green"),
    #     ("This is not correct.", "red"),
    # ]

    img = mpimg.imread(abs_img_path)
    cls_name = abs_img_path.rsplit('/', 2)[-2]

    # Create a figure with a grid for the image and the text
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])  # Two rows, one for image and one for caption

    # Plot the image on the left (top row)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis('off')  # Hide axes for the image

    # Add a caption below the image (bottom row)
    ax_caption = fig.add_subplot(gs[1, 0])
    ax_caption.text(0.5, 0.5, cls_name, fontsize=12, ha='center', va='center')
    ax_caption.axis('off')  # Hide axes for the caption

    # Plot the sentences on the right (top row)
    ax_text = fig.add_subplot(gs[0:, 1])
    ax_text.axis("off")  # Hide axes for the text

    # Plot each sentence on a different row with bounding boxes
    for i, (sentence, color) in enumerate(sentences):
        ax_text.text(0.1, 1 - (i + 1) * 0.2, sentence, fontsize=12, color='black', va='center',
                    bbox=dict(facecolor='none', edgecolor=color, boxstyle='round,pad=0.5', linewidth=2))

    # Adjust layout and show plot
    plt.tight_layout()
    os.makedirs(os.path.join(plotting_dir, 'sentences_plots'), exist_ok=True)
    img_name = abs_img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    plt.savefig(os.path.join(plotting_dir, 'sentences_plots', f"sentences_plot_{sample_id}_{img_name}.png"), dpi=600, bbox_inches='tight')
    # plt.show()

def plot_concept_attn_score_all_samples():
    # Example sentences with corresponding colors (green for positive, red for negative)
    sentences = [
        ("This is a good sentence.", "green"),
        ("This is a bad sentence.", "red"),
        ("Everything is fine here.", "green"),
        ("This is not correct.", "red"),
    ]

    # Create a new figure
    fig, ax = plt.subplots()

    # Remove axes
    ax.axis("off")

    # Plot each sentence on a different row with appropriate color
    for i, (sentence, color) in enumerate(sentences):
        ax.text(0.1, 1 - (i + 1) * 0.1, sentence, fontsize=12, color=color, va='center')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()