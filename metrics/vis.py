import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt


def plot_concept_attn_score_single_sample(gt_concepts_raw, pred_concepts_raw):
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