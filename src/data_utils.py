"""
Pomoćne funkcije za rad sa MedicaLeaf datasetom:
- kreiranje tabele (DataFrame) sa putanjama do slika i tipom lista
- prikaz raspodele broja slika po klasama
- kreiranje augmentovanog trening skupa na disku
"""

import os

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

def save_augmented_dataset(df, output_dir, transform, target_count=160):
    """
    Cuva augemntisane slike u direktorijum
    transform - transformacija koja se primenjuje na slike
    """
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Augmentacija već urađena u '{output_dir}', preskačem.")
        return

    resize = transforms.Resize((224, 224)) 
    print("Počinjem augmentaciju...")
    for leaf_type, group in df.groupby("leaf_type"):
        class_dir = os.path.join(output_dir, leaf_type)
        os.makedirs(class_dir, exist_ok=True)

        imgs = group["image_path"].tolist()
        current = len(imgs)

        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
            img.save(os.path.join(class_dir, os.path.basename(img_path)))

        if current >= target_count:
            continue

        needed = target_count - current
        generated = 0
        while generated < needed:
            src = imgs[generated % current]
            img = Image.open(src).convert("RGB")
            aug_img = transform(img)
            filename = f"aug_{generated:04d}_{os.path.basename(src)}"
            aug_img.save(os.path.join(class_dir, filename))
            generated += 1

    print(f"\nGotovo! Slike sačuvane u '{output_dir}'.")

def build_dataframe(data_dir):
    """
    Kreira DataFrame sa putanjama do slika i tipom lista.
    """
    image_paths = []
    leaf_types = []

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        for image_name in sorted(os.listdir(folder_path)):
            image_paths.append(os.path.join(folder_path, image_name))
            leaf_types.append(folder)

    return pd.DataFrame({"image_path": image_paths, "leaf_type": leaf_types})


def plot_class_distribution(df, title="Distribucija broja slika po klasama"):
    """
    Plotuje raspodelu broja po klasama u DataFrame-u.
    """
    counts = df["leaf_type"].value_counts()

    print("Ukupan broj slika:", len(df))

    plt.figure(figsize=(18, 8))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Tip lista")
    plt.ylabel("Broj slika")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()


def show_sample_images(df, n=5):
    """
    Prikazuje n nasumicnih slika
    """
    sample_images = df.sample(n)

    plt.figure(figsize=(4 * n, 4))
    for i, (index, row) in enumerate(sample_images.iterrows()):
        img = Image.open(row["image_path"])
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(row["leaf_type"])
        plt.axis("off")
    plt.show()
