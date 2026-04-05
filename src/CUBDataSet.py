from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CUBDataSet:
    # split: 1 --> train
    #       0 --> test

    def __init__(self, split, samples):
        if split == "train":
            self.is_train = 1
        else:
            self.is_train = 0
        self.samples = samples
        self.transforms = build_image_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample.get("label")
        path = sample.get("path")
        with Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transforms(img), label)
