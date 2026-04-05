
try:
    from .dataset import DataSet
    from .model import EmbeddingModel as Model
    from .config import Config
except ImportError:
    from dataset import DataSet
    from model import EmbeddingModel as Model
    from config import Config
    

import csv
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
import yaml

EVAL_LOG_FILENAME = "evaluate_log_v2.csv"


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
    print(f"Archived incompatible evaluate log to: {backup_path}")
    return True


def load_evaluate_config(run_config_path: Path | None) -> Config:
    if run_config_path is None or not run_config_path.exists():
        return Config()

    cfg = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    requested_device = cfg["runtime"]["device"]

    if requested_device == "cuda" and not torch.cuda.is_available():
        cfg["runtime"]["device"] = "auto"
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        cfg["runtime"]["device"] = "auto"

    if cfg["runtime"]["device"] == requested_device:
        return Config(run_config_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        temp_path = Path(tmp.name)

    try:
        print(
            f"Runtime device '{requested_device}' is unavailable on this machine. "
            "Falling back to auto for evaluation."
        )
        return Config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)



class Evaluate:

    def __init__(self, checkpoint_path,save_fail=False,dist_matrix=None):
        self.path=Path(checkpoint_path)
        run_config_path = self.path.parent / "config.yaml"
        self.config=load_evaluate_config(run_config_path)
        self.checkpoint_epoch = None
        self.model=self.load_checkpoint()
        self.test_dataset=DataSet(self.config.root).test_list
        self.dataloader=self.get_dataloader()
        self.test_embedding_list,self.labels_list=self.get_test_embedding_label_list()
        self.save_fail=save_fail
        self.dist_matrix=dist_matrix
        
    def load_checkpoint(self):
        
        checkpoint=torch.load(self.path, map_location=self.config.device)
        state_dict = checkpoint["model_state_dict"]
        self.embed_dim = state_dict["embedding_head.2.weight"].shape[0]
        self.checkpoint_epoch = checkpoint.get("epoch")
        
        model=Model(self.embed_dim)
        model.to(self.config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    
    def get_dataloader(self):

        dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers

        )
        return dataloader
        
   
    def get_test_embedding_label_list(self):
        
        return get_test_embedding_label_list(self.config,self.model,self.dataloader)
    
    

    def get_dist_matrix(self):
      
        return get_dist_matrix(self.test_embedding_list)




    def recall_at_k(self,k=5):
        if self.dist_matrix is None:
            self.dist_matrix=self.get_dist_matrix()
        
        result, fail_case=recall_at_k(self.test_dataset.samples,self.dist_matrix,k)

        if self.save_fail:
            self.log_failure_cases(self.path, self.config.results_dir, k, fail_case)

        return result
    

    def mAP(self):
        
        return mAP(self.test_embedding_list,self.labels_list,self.dist_matrix)
    
    def evaluate(self):

        print("evaluating.....")
        recall1=self.recall_at_k(1)
        recall5=self.recall_at_k(5)
        recall10=self.recall_at_k(10)
        mAP=self.mAP()

        print(f"Evaluation For: {self.path}")

        print("+------------+----------+")
        print(f"| {'Metric':<10} | {'Value':<8} |")
        print("+------------+----------+")
        print(f"| {'Recall@1':<10} | {recall1:<8.4f} |")
        print(f"| {'Recall@5':<10} | {recall5:<8.4f} |")
        print(f"| {'Recall@10':<10} | {recall10:<8.4f} |")
        print(f"| {'mAP':<10} | {mAP:<8.4f} |")
        print("+------------+----------+")
        self.log_experiment(recall1, recall5, recall10, mAP)

    def log_experiment(self, recall1, recall5, recall10, mAP):
        log_path = self.config.results_dir / EVAL_LOG_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": str(self.path),
            "run_id": self.path.parent.name,
            "epoch": self.checkpoint_epoch if self.checkpoint_epoch is not None else self.path.stem,
            "loss_method": self.config.loss_method,
            "embedding_dim": self.embed_dim,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "recall@1": f"{recall1:.6f}",
            "recall@5": f"{recall5:.6f}",
            "recall@10": f"{recall10:.6f}",
            "mAP": f"{mAP:.6f}",
        }

        fieldnames = list(row.keys())
        write_header = ensure_csv_schema(log_path, fieldnames)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
          
    def log_failure_cases(self,path,results_dir,k,fail_case):
        checkpoint_path = Path(path)
        log_path = results_dir / checkpoint_path.parent.name / f"{checkpoint_path.stem}_fail_cases_recall_at_{k}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["checkpoint", "fail_case", "recall@"]
        write_header = ensure_csv_schema(log_path, fieldnames)

        with log_path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for fail_path in fail_case:
                row={
                    "checkpoint": str(self.path),
                    "fail_case": fail_path,
                    "recall@" : k
                }
                writer.writerow(row)
        
        print(f"Fail cases have been stored in {log_path}")


               
      
def run_recall_at_k(config,model,k):
        test_dataset=DataSet(config.root).test_list
        dataloader=get_dataloader(test_dataset,config)
        test_samples=test_dataset.samples
        test_embedding_list,labels_list=get_test_embedding_label_list(config,model,dataloader)       
        dist_matrix=get_dist_matrix(test_embedding_list)
        return recall_at_k(test_samples,dist_matrix,k)


def recall_at_k(test_samples,distable,k=5):


    N= len(test_samples)

    success_count=0
    fail_case=[]
    for idx in range(N):
        # Distances from image with idx to others inside the test set
        dist_row= distable[idx]
        _,indeces=torch.topk(dist_row,k,largest=False)

        query_label=test_samples[idx].get("label")

        is_success=False
        for oidx in indeces:
            if query_label==test_samples[oidx].get("label"):
                success_count+=1
                is_success=True
                break
        
        if not is_success:
            fail_case.append(test_samples[idx].get("path"))

    return [success_count / N, fail_case]

def run_mAP(config,model):
        test_dataset=DataSet(config.root).test_list
        dataloader=get_dataloader(test_dataset,config)
        
        test_embedding_list,labels_list=get_test_embedding_label_list(config,model,dataloader)       
        dist_matrix=get_dist_matrix(test_embedding_list)
        return mAP(test_embedding_list,labels_list,dist_matrix)

def mAP(test_embedding_list,labels_list,dist_matrix):
        
    

        N=len(test_embedding_list)
        total_AP=0.0
        valid_query=0
        for i in range(N):


            qury_label=labels_list[i]
            dist_list=dist_matrix[i]
            label_mask=(labels_list==qury_label).int()
            label_mask[i]=0
            relevent_items = label_mask.sum().item()

            if relevent_items==0:
                continue
            
            valid_query+=1
            sorted_indices = torch.argsort(dist_list)


            current_relevent=0
            query_AP=0.0
            for s_idx in range(N):

                current_label= label_mask[sorted_indices[s_idx]]
                
                if current_label==1:
                    current_relevent+=1
                    query_AP+=current_relevent / (s_idx+1)
                
                if current_relevent== relevent_items:
                    break
            


            total_AP+= query_AP / relevent_items
        
        mAP= total_AP / valid_query
        return mAP


def get_dataloader(test_dataset,config):

    dataloader=DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers

    )
    return dataloader


def get_test_embedding_label_list(config,model,dataloader):
    all_embeddings=[]
    all_labels=[]
    with torch.no_grad():

        for images, labels in dataloader:
            images=images.to(config.device)
            embeddings=model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    all_embeddings=torch.cat(all_embeddings,dim=0)
    all_labels=torch.cat(all_labels,dim=0)
    
    return [all_embeddings,all_labels]

def get_dist_matrix(test_embedding_list):


    sq_norm=torch.sum(test_embedding_list**2,dim=1)

    dist_sq=sq_norm.unsqueeze(1)+sq_norm.unsqueeze(0)-(test_embedding_list@test_embedding_list.T)*2

    distable=torch.sqrt(torch.clamp(dist_sq,min=1e-12))

    distable.fill_diagonal_(1e9)

    return distable

                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_fail", action="store_true")
    return parser.parse_args()

        



def main():
    args = parse_args()
    path = args.checkpoint
    eval=Evaluate(path,save_fail=args.save_fail)
    eval.evaluate()


if __name__ =="__main__":
    main()
            
