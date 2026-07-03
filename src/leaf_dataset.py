

from PIL import Image
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """
    Dataset za slike listova.
    """
    def __init__(self, dataframe, transform=None, class_to_idx=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(self.dataframe["leaf_type"].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.class_to_idx = class_to_idx
        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[row["leaf_type"]]
        return image, label
