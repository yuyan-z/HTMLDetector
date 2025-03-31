import math
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib import patches
from PIL import Image


# DATASET_PATH = os.path.join(os.getcwd(), 'datasets')
DATASET_PATH = "datasets"

def load_classes() -> list:
    """
    Load class with a name and a color.
    :return: A list of dicts, each dict contains
             - 'name': class name (str)
             - 'rgba': RGBA color (tuple of floats)
    """
    yaml_path = os.path.join(DATASET_PATH, "data.yaml")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    classes = data['names']
    # print(len(classes))  # 18

    colormap = plt.get_cmap('tab20', len(classes))
    classes_with_color = [{"name": cls, "rgba": colormap(i)} for i, cls in enumerate(classes)]

    return classes_with_color


def load_labels_df(split: str) -> pd.DataFrame:
    """
    Load labels data from YOLO-format text files and return as a pandas DataFrame.
    :param split: str. "train", "valid" or "test"
    :return: A pandas DataFrame
    """
    labels_dir = os.path.join(DATASET_PATH, split, "labels")
    data = []
    for filename in os.listdir(labels_dir):
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                class_id, x, y, w, h = line.strip().split()
                data.append({
                    "image_id": filename.replace(".txt", ""),
                    "class_id": int(class_id),
                    "x_center": float(x),
                    "y_center": float(y),
                    "width": float(w),
                    "height": float(h)
                })

    df = pd.DataFrame(data)

    return df


def plot_images(image_ids: list, df: pd.DataFrame, images_dir: str, classes: list, cols: int = 4) -> None:
    """
    Plot images with bounding boxes using matplotlib.
    """
    rows = math.ceil(len(image_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        image = Image.open(image_path)
        w, h = image.size
        labels_df = df[df['image_id'] == image_id]

        ax = axes[i]
        ax.imshow(image)
        title = image_id.split(".")[0]
        ax.set_title(f"image_id={title}")
        ax.axis('on')
        padding = 10
        ax.set_xlim([-padding, w + padding])
        ax.set_ylim([h + padding, -padding])

        for row in labels_df.itertuples():
            x1 = int((row.x_center - row.width / 2) * w)
            y1 = int((row.y_center - row.height / 2) * h)
            box_width = row.width * w
            box_height = row.height * h
            rgba = classes[int(row.class_id)]["rgba"]
            name = classes[int(row.class_id)]["name"]
            rect = patches.Rectangle((x1, y1), box_width, box_height,
                                     linewidth=2, edgecolor=rgba, facecolor='none')
            ax.add_patch(rect)

            ax.text(x1, y1 - 5, name, color=rgba, fontsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def sample_image_paths(split: str, n: int):
    images_dir = os.path.join(DATASET_PATH, split, "images")
    filenames = [f for f in os.listdir(images_dir)]
    if len(filenames) < n:
        n = len(filenames)
    filenames = random.sample(filenames, n)
    paths = [os.path.join(images_dir, f) for f in filenames]
    return paths


if __name__ == "__main__":
    classes = load_classes()
    # print(classes)

    train_df = load_labels_df("train")
    # print(train_df.head())

    n_samples = 8
    image_ids = list(train_df["image_id"].drop_duplicates().sample(n=n_samples, random_state=1).values)
    samples_df = train_df[train_df["image_id"].isin(image_ids)]
    # print(samples_df)

    train_images_dir = os.path.join(DATASET_PATH, "train", "images")
    plot_images(image_ids, samples_df, train_images_dir, classes)

    image_paths = sample_image_paths("test", 10)
    print(image_paths)

