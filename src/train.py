try:
    from .dataset import DataSet, ensure_cub_dataset_available
    from .loss import MyTripletMarginLoss as TripletLoss, MyProxyNCA as ProxyNCALoss
    from .model import EmbeddingModel as Model
    from .pk_sampler import PK_Sampler
    from .config import Config
    from .evaluate import Evaluate, run_mAP
except ImportError:
    from dataset import DataSet, ensure_cub_dataset_available
    from loss import MyTripletMarginLoss as TripletLoss, MyProxyNCA as ProxyNCALoss
    from model import EmbeddingModel as Model
    from pk_sampler import PK_Sampler
    from config import Config
    from evaluate import Evaluate, run_mAP

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
from pathlib import Path
from datetime import datetime
import argparse
import csv
import yaml

TRAIN_LOG_FILENAME = "train_log_v2.csv"


def ensure_csv_schema(log_path: Path, fieldnames: list[str]) -> bool:
    if not log_path.exists():
        return True

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)

    if existing_header == fieldnames:
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = log_path.with_name(f"{log_path.stem}_legacy_{timestamp}{log_path.suffix}")
    log_path.replace(backup_path)
    print(f"Archived incompatible train log to: {backup_path}")
    return True


def train_one_epoch(config,model: Model, optimizer,loss, train_loader, epoch):

    num_epoches=config.epochs
    device=config.device
    log_every_steps=config.log_every_steps

    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"train-epoch {epoch}/{num_epoches}")
    for step, (images, labels) in enumerate(progress_bar, start=1):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        train_embeddings = model(images)
        train_loss = loss(train_embeddings, labels)
        train_loss.backward()
        optimizer.step()
        loss_value = train_loss.item()
        total_loss += loss_value
        if step % log_every_steps == 0 or step == len(train_loader):
            progress_bar.set_postfix(loss=f"{loss_value:.4f}", avg=f"{total_loss / step:.4f}")
    
    return total_loss / len(train_loader)


def build_training_components(config):
    model = Model(config.embedding_dim).to(config.device)

    if config.loss_method == "proxyNCA":
        loss = ProxyNCALoss(config.classes_num, config.embedding_dim).to(config.device)
    else:
        loss = TripletLoss()

    optimizer = build_optimizer(config, model, loss, config.loss_method)
    scheduler = build_scheduler(config, optimizer)
    return model, loss, optimizer, scheduler


def build_optimizer(config, model: Model, loss, loss_method: str):
    param_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": config.backbone_learning_rate
            if config.backbone_learning_rate is not None
            else config.learning_rate,
        },
        {
            "params": model.embedding_head.parameters(),
            "lr": config.head_learning_rate
            if config.head_learning_rate is not None
            else config.learning_rate,
        },
    ]
    if loss_method == "proxyNCA":
        param_groups.append(
            {
                "params": loss.parameters(),
                "lr": config.proxy_learning_rate
                if config.proxy_learning_rate is not None
                else config.learning_rate,
            }
        )
    return optim.Adam(param_groups, weight_decay=config.weight_decay)


def build_scheduler(config, optimizer):
    scheduler_name = config.scheduler
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        if config.scheduler_step_size is None:
            raise ValueError("train.scheduler_step_size must be set when train.scheduler is 'step'.")
        return lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.scheduler_min_lr,
        )
    raise ValueError(f"Unsupported train.scheduler: {scheduler_name}")


def optimizer_lrs(optimizer):
    lrs = [group["lr"] for group in optimizer.param_groups]
    backbone_lr = lrs[0] if len(lrs) > 0 else 0.0
    head_lr = lrs[1] if len(lrs) > 1 else backbone_lr
    proxy_lr = lrs[2] if len(lrs) > 2 else head_lr
    return backbone_lr, head_lr, proxy_lr


def validate_one_epoch(config,model:Model,epoch)->float:
    num_epoches=config.epochs

    model.eval()

    map= run_mAP(config,model)

    print(f"epoch {epoch}/{num_epoches} mAP: {map}")
    model.train()
    return map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str,default=None)
    parser.add_argument("--epochs", type=int,default=None)
    parser.add_argument("--download-data", action="store_true")

    return parser.parse_args()


def log_train_result(
    config,
    run_id,
    epoch,
    train_loss,
    map,
    weight_decay,
    best_map,
    checkpoint_updated,
    optimizer,
):
    log_path = config.results_dir / TRAIN_LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": run_id,
        "epoch": epoch,
        "loss_method": config.loss_method,
        "embedding_dim": config.embedding_dim,
        "learning_rate": config.learning_rate,
        "backbone_learning_rate": (
            config.backbone_learning_rate
            if config.backbone_learning_rate is not None
            else config.learning_rate
        ),
        "head_learning_rate": (
            config.head_learning_rate
            if config.head_learning_rate is not None
            else config.learning_rate
        ),
        "proxy_learning_rate": (
            config.proxy_learning_rate
            if config.proxy_learning_rate is not None
            else config.learning_rate
        ),
        "scheduler": config.scheduler,
        "scheduler_step_size": config.scheduler_step_size,
        "scheduler_gamma": f"{config.scheduler_gamma:.6f}",
        "scheduler_min_lr": f"{config.scheduler_min_lr:.6f}",
        "epochs": config.epochs,
        "margin": config.margin,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "sampler_p": config.sampler_p,
        "sampler_k": config.sampler_k,
        "seed": config.seed,
        "optimizer_backbone_lr": f"{optimizer_lrs(optimizer)[0]:.6f}",
        "optimizer_head_lr": f"{optimizer_lrs(optimizer)[1]:.6f}",
        "optimizer_proxy_lr": f"{optimizer_lrs(optimizer)[2]:.6f}",
        "train_loss": f"{train_loss:.6f}",
        "mAP": f"{map:.6f}",
        "best_mAP_so_far": f"{best_map:.6f}",
        "checkpoint_updated": checkpoint_updated,
        "weight_decay": f"{weight_decay:.6f}"
    }

    fieldnames = list(row.keys())
    write_header = ensure_csv_schema(log_path, fieldnames)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_config_snapshot(config, save_path):
    config.cfg["train"]["epochs"] = config.epochs
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.cfg, f, sort_keys=False)


def load_config_from_args(args):
    if args.config is not None:
        return Config(args.config)
    return Config()


def resolve_num_epochs(config, override_epochs):
    if override_epochs is None:
        return config.epochs
    config.epochs = override_epochs
    return override_epochs


def resume_training_state(
    resume_checkpoint_path,
    device,
    model,
    optimizer,
    scheduler,
    loss,
    loss_method,
    config,
):
    epoch_num_bf = 0
    best_map = float("-inf")
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if resume_checkpoint_path is None:
        return epoch_num_bf, best_map, start_time

    checkpoint = torch.load(resume_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_map = checkpoint.get("best_map", float("-inf"))
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if loss_method == "proxyNCA":
        loss.load_state_dict(checkpoint["loss_state_dict"])

    epoch_num_bf = checkpoint["epoch"]
    config.epochs += epoch_num_bf
    start_time = Path(resume_checkpoint_path).parent.name
    return epoch_num_bf, best_map, start_time


def prepare_run_dirs(project_root, checkpoint_dir, run_id):
    run_checkpoint_dir = checkpoint_dir / run_id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = project_root / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_checkpoint_dir, results_dir


def build_train_loader(train_set, is_debug, sampler_p, sampler_k, num_workers):
    train_pk_sampler = PK_Sampler(train_set, is_debug, sampler_p, sampler_k)
    return DataLoader(
        train_set,
        batch_sampler=train_pk_sampler,
        num_workers=num_workers,
    )


def save_best_checkpoint(best_checkpoint_path, epoch, best_map, model, optimizer, scheduler, loss, loss_method):
    checkpoint = {
        "epoch": epoch,
        "best_map": best_map,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if loss_method == "proxyNCA":
        checkpoint["loss_state_dict"] = loss.state_dict()
    torch.save(checkpoint, best_checkpoint_path)


def run_training_loop(
    config,
    model,
    optimizer,
    loss,
    scheduler,
    train_loader,
    run_id,
    start_epoch,
    num_epochs,
    best_map,
    best_checkpoint_path,
):
    train_loss_history = []
    map_history = []

    for epoch_offset in range(num_epochs):
        epoch = epoch_offset + 1 + start_epoch
        train_epoch_loss = train_one_epoch(
            config,
            model,
            optimizer,
            loss,
            train_loader,
            epoch,
        )
        train_loss_history.append(train_epoch_loss)
        print(f"train epoch {epoch} avg loss: {train_epoch_loss:.4f}")

        epoch_map = validate_one_epoch(config, model, epoch)
        map_history.append(epoch_map)
        print(f"evaluate epoch {epoch} mAP: {epoch_map:.4f}")

        improved = False
        if epoch_map > best_map:
            best_map = epoch_map
            improved = True
            save_best_checkpoint(
                best_checkpoint_path,
                epoch,
                best_map,
                model,
                optimizer,
                scheduler,
                loss,
                config.loss_method,
            )
            print(f"saved improved checkpoint: {best_checkpoint_path} (best mAP={best_map:.4f})")

        log_train_result(
            config=config,
            run_id=run_id,
            epoch=epoch,
            train_loss=train_epoch_loss,
            map=epoch_map,
            weight_decay=config.weight_decay,
            best_map=best_map,
            checkpoint_updated=improved,
            optimizer=optimizer,
        )
        if scheduler is not None:
            scheduler.step()

    return train_loss_history, map_history, best_map


def plot_training_curves(results_dir, run_id, start_epoch, num_epochs, train_loss_history, map_history):
    import matplotlib.pyplot as plt

    x_axis = range(1 + start_epoch, num_epochs + 1 + start_epoch)
    plt.plot(x_axis, train_loss_history, label="Train", color="orange")
    plt.plot(x_axis, map_history, label="mAP", color="blue")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title(f"Training Loss VS mAP({run_id})")
    plt.savefig(results_dir / f"{run_id}_{start_epoch}_loss_curve.png")
    plt.close()

def main():
    args = parse_args()
    config = load_config_from_args(args)
    is_debug = args.debug
    resume_checkpoint_path = args.resume

    config.set_seed()
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    project_root = config.project_root
    device = config.device
    num_epochs = resolve_num_epochs(config, args.epochs)
    data_root = ensure_cub_dataset_available(config.root, download=args.download_data)

    model, loss, optimizer, scheduler = build_training_components(config)
    epoch_num_bf, best_map, start_time = resume_training_state(
        resume_checkpoint_path,
        device,
        model,
        optimizer,
        scheduler,
        loss,
        config.loss_method,
        config,
    )

    run_checkpoint_dir, results_dir = prepare_run_dirs(project_root, checkpoint_dir, start_time)
    save_config_snapshot(config, run_checkpoint_dir / "config.yaml")
    save_config_snapshot(config, results_dir / "config.yaml")

    # set dataset
    dataset = DataSet(data_root)
    train_set = dataset.train_list
    test_set = dataset.test_list

    # train
    print(f"device: {device}")
    print(f"train samples: {len(train_set)}")
    print(f"test samples: {len(test_set)}")
    print(f"scheduler: {config.scheduler}")

    train_loader = build_train_loader(
        train_set,
        is_debug,
        config.sampler_p,
        config.sampler_k,
        config.num_workers,
    )
    best_checkpoint_path = run_checkpoint_dir / "best.pth"
    train_loss_history, map_history, best_map = run_training_loop(
        config,
        model,
        optimizer,
        loss,
        scheduler,
        train_loader,
        start_time,
        epoch_num_bf,
        num_epochs,
        best_map,
        best_checkpoint_path,
    )
    plot_training_curves(
        results_dir,
        start_time,
        epoch_num_bf,
        num_epochs,
        train_loss_history,
        map_history,
    )


if __name__ == "__main__":
    main()
