from pathlib import Path
import random
import numpy as np
import torch
import yaml

class Config:

    def __init__(self,config_path=Path(__file__).resolve().parent.parent / "configs/default.yaml"):
        self.config_path=config_path
        self.cfg=self.load_config()
        self.root=self.cfg["data"]["root"]
        self.image_size=self.cfg["data"]["image_size"]
        self.num_workers=self.cfg["data"]["num_workers"]
        self.classes_num=self.cfg["data"]["classes_num"]
        
        self.batch_size=self.cfg["train"]["batch_size"]
        self.learning_rate=self.cfg["train"]["learning_rate"]
        self.backbone_learning_rate=self.cfg["train"].get("backbone_learning_rate")
        self.head_learning_rate=self.cfg["train"].get("head_learning_rate")
        self.proxy_learning_rate=self.cfg["train"].get("proxy_learning_rate")
        self.scheduler=self.cfg["train"].get("scheduler", "none")
        self.scheduler_step_size=self.cfg["train"].get("scheduler_step_size")
        self.scheduler_gamma=float(self.cfg["train"].get("scheduler_gamma", 0.1))
        self.scheduler_min_lr=float(self.cfg["train"].get("scheduler_min_lr", 1e-6))
        self.epochs=self.cfg["train"]["epochs"]
        self.margin=self.cfg["train"]["margin"]
        self.seed=self.cfg["train"]["seed"]
        self.sampler_k=self.cfg["train"]["sampler_k"]
        self.sampler_p=self.cfg["train"]["sampler_p"]
        self.loss_method=self.cfg["train"]["loss_method"]
        self.weight_decay=float(self.cfg["train"]["weight_decay"])

        self.embedding_dim=self.cfg["model"]["embedding_dim"]

        self.project_root=Path(__file__).resolve().parent.parent
        self.checkpoint_dir=self.project_root / self.cfg["paths"]["checkpoint_dir"]
        self.results_dir=self.project_root / self.cfg["paths"]["results_dir"]

        self.device=self._resolve_device(self.cfg["runtime"]["device"])
        self.log_every_steps=self.cfg["runtime"]["log_every_steps"]
        
        
        

    def load_config(self):
        config_path=self.config_path
        with Path(config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def set_seed(self):
        seed=self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _resolve_device(self, requested_device):
        if requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False.")
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Config requested MPS but torch.backends.mps.is_available() is False.")
        if requested_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported runtime.device: {requested_device}")
        return requested_device
