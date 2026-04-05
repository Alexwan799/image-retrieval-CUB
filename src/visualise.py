try:
    from .dataset import DataSet
    from .model import EmbeddingModel as Model
    from .evaluate import Evaluate
    from .CUBDataSet import build_image_transform
except ImportError:
    from dataset import DataSet
    from model import EmbeddingModel as Model
    from evaluate import Evaluate
    from CUBDataSet import build_image_transform

import torch
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse

def _get_model_device(model: Model) -> torch.device:
    return next(model.parameters()).device

def get_query_embedding(model:Model,path):

    trans = build_image_transform()

    with Image.open(path) as img:
        img=img.convert("RGB")
    img=trans(img)
    img=img.unsqueeze(0)
    model_device = _get_model_device(model)
    img=img.to(model_device)
    with torch.no_grad():
        embedding=model(img)

    return embedding.detach().cpu()

# return [idx]
def get_top_k(query_embedding,test_embedding_list,k=5):

    q= query_embedding.squeeze(0).cpu()
    t=test_embedding_list.cpu()
    q_norm=torch.sum(q**2)
    t_norm=torch.sum(t **2,dim=1)
    dot=t @ q
    dist_list=q_norm+t_norm- 2*dot
    dist_list=torch.sqrt(torch.clamp(dist_list,min=0.0))

    candidate_k = min(k + 20, dist_list.numel())
    values,indeces=torch.topk(dist_list,candidate_k,largest=False)
    
    mask=values > 1e-3
    values = values[mask][:k]
    indeces= indeces[mask][:k]


    return [indeces,values]


def visualise(query_img_path,result_idxes,result_dists,test_samples,query_label_id,query_label_name):

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "results" / "query_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_save_path = output_dir / f"{start_time}.png"

    results=[]

    for idx,dist in zip(result_idxes,result_dists):
        d=test_samples[idx]
        d["dist"]=dist
        results.append(d)
    

    # Load names of the labels
    label_name_dict={}

    data_root = project_root / "data" / "CUB_200_2011"
    with (data_root / "classes.txt").open("r", encoding="utf-8") as label_f:
        for line in label_f:
            line = line.strip()
            if not line:
                continue
            left, right = line.split(" ", 1)
            label_name_dict[int(left)]=right



    fig,axes=plt.subplots(1,len(results)+1, figsize=(18,4))
    query_img=Image.open(query_img_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query img \n label id: {query_label_id} \n label name: {query_label_name}")
    axes[0].axis("off")

    for i in range( len(results)):
        d=results[i]
        path=d["path"]
        label=d["label"]
        dist=d["dist"]
        label_name=label_name_dict[label+1]
        result_img=Image.open(path).convert("RGB")
        axes[i+1].imshow(result_img)
        axes[i+1].set_title(f"Top {i+1} \n label id: {label+1} \n label name: {label_name} \n dist: {dist:.3f} ")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(img_save_path)
    plt.show()



    
def query(query_img_path,checkpoint_path,query_num,eval=None):

    if eval is None:
        eval=Evaluate(checkpoint_path)
    query_label_id, query_label_name= infer_query_label_from_path(query_img_path)
    test_embedding_list=eval.test_embedding_list
    model=eval.model
    query_embedding=get_query_embedding(model,query_img_path)

    result_idxes,result_dists=get_top_k(query_embedding,test_embedding_list,query_num)

    visualise(query_img_path,result_idxes,result_dists,eval.test_dataset.samples,query_label_id,query_label_name)

def infer_query_label_from_path(query_img_path):
    parent_name = Path(query_img_path).parent.name

    if "." not in parent_name:
        return None, "unknown"

    label_id, label_name = parent_name.split(".", 1)

    if not label_id.isdigit():
        return None, "unknown"

    return int(label_id), label_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--q_image", type=str,default=None)
    parser.add_argument("--q_num", type=int,default=None)


    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path=args.checkpoint
    query_img_path=args.q_image
    query_num=args.q_num
    if query_num is None:
        query_num=5

    query(query_img_path,checkpoint_path,query_num)




if __name__=="__main__":
    main()
