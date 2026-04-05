# DataSet: initiate an object based on the img_id
# function:
# 1. 读 images.txt（image_id -> 相对路径）
# 2. 读 image_class_labels.txt（image_id -> class_id）
# 3. 读 train_test_split.txt（image_id -> train/test）
# 4. 按 split=train/test 过滤出样本列表
# 5. 在 getitem 里：
#    用 PIL 打开图片
#   转 RGB
#   做 transform（resize/to tensor/normalize）
#   返回图像和标签
from pathlib import Path
import tarfile
import urllib.request

DEFAULT_CUB_DOWNLOAD_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
]
REQUIRED_DATASET_FILES = (
    "images.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "classes.txt",
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_data_root(data_root=None) -> Path:
    if data_root is None:
        return get_project_root() / "data" / "CUB_200_2011"

    data_root = Path(data_root)
    if data_root.is_absolute():
        return data_root
    return get_project_root() / data_root


def is_cub_dataset_available(data_root=None) -> bool:
    data_root = resolve_data_root(data_root)
    if not data_root.exists():
        return False
    if not (data_root / "images").is_dir():
        return False
    return all((data_root / name).exists() for name in REQUIRED_DATASET_FILES)


def download_and_extract_cub_dataset(data_root=None, download_urls=None) -> Path:
    data_root = resolve_data_root(data_root)
    data_root.parent.mkdir(parents=True, exist_ok=True)

    if is_cub_dataset_available(data_root):
        return data_root

    archive_path = data_root.parent / "CUB_200_2011.tgz"
    urls = download_urls or DEFAULT_CUB_DOWNLOAD_URLS
    last_error = None

    for url in urls:
        try:
            print(f"Downloading CUB-200-2011 from {url}")
            urllib.request.urlretrieve(url, archive_path)
            break
        except Exception as exc:
            last_error = exc
            if archive_path.exists():
                archive_path.unlink()
    else:
        raise RuntimeError("Failed to download CUB-200-2011 from all known URLs.") from last_error

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_root.parent)
    finally:
        archive_path.unlink(missing_ok=True)

    if not is_cub_dataset_available(data_root):
        raise RuntimeError(f"Downloaded archive but dataset layout under '{data_root}' is incomplete.")

    return data_root


def ensure_cub_dataset_available(data_root=None, download=False) -> Path:
    data_root = resolve_data_root(data_root)
    if is_cub_dataset_available(data_root):
        return data_root

    if download:
        return download_and_extract_cub_dataset(data_root)

    raise FileNotFoundError(
        f"CUB-200-2011 dataset not found at '{data_root}'. "
        "Place the dataset there or re-run with '--download-data'."
    )


def load_cub_dataset_class():
    try:
        from .CUBDataSet import CUBDataSet
    except ImportError:
        from CUBDataSet import CUBDataSet
    return CUBDataSet


class DataSet:
    def __init__(self, data_root=None):
        self.data_root = ensure_cub_dataset_available(data_root, download=False)
        train_samples, test_samples = self.load_metadata()
        CUBDataSet = load_cub_dataset_class()
        self.train_list = CUBDataSet("train", train_samples)
        self.test_list = CUBDataSet("test", test_samples)

    def load_metadata(self):
        data_root = self.data_root

        label_id = {}
        with (data_root / "image_class_labels.txt").open("r", encoding="utf-8") as label_f:
            for line in label_f:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split(" ", 1)
                label_id[left] = right

        path_prefix = data_root / "images"
        path = {}
        with (data_root / "images.txt").open("r", encoding="utf-8") as path_f:
            for line in path_f:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split(" ", 1)
                path[left] = str(path_prefix / right)

        is_train = {}
        with (data_root / "train_test_split.txt").open("r", encoding="utf-8") as train_f:
            for line in train_f:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split(" ", 1)
                is_train[left] = right

        if len(label_id) != len(path) or len(path) != len(is_train):
            raise ValueError(
                "The number of data don't match between classes.txt, image_class_labels.txt, images.txt and train_test_split.txt!"
            )

        train_samples = []
        test_samples = []

        for k, v in path.items():
            d = {}
            d["is_train"] = is_train.get(k)
            d["label"] = int(label_id.get(k)) - 1
            d["path"] = v
            d["img_id"] = k
            if is_train.get(k) == "1":
                train_samples.append(d)
            else:
                test_samples.append(d)

        return (train_samples, test_samples)
