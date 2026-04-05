from collections import defaultdict
import random as rand
class PK_Sampler:
    # p: how many classes inside one batch
    # k: how many sampels inside one class of the batch
    def __init__(self,train_test_list,is_debug,p=8,k=4):
        self.p=p
        self.k=k

        if is_debug:
            self.num_batches=20
        else:
            self.num_batches=len(train_test_list) // (self.p * self.k)
        
        self.cubdataset=train_test_list
        self.sample_diclist=train_test_list.samples
        self.label_groups=self.get_label_group()
        self.labels=list(self.label_groups.keys())
        if p>len(self.labels):
            raise ValueError("The value of P is too large!")
    
    def get_label_group(self):

        labels2idx=defaultdict(list)
        length=len(self.sample_diclist)
        for i in range(length):
            d=self.sample_diclist[i]
            label=d.get("label")
            labels2idx[label].append(i)
        return labels2idx
    

    def get1batch(self):


        P_labels= rand.sample(self.labels,self.p)
        batch_list=[]

        for label in P_labels:
            K_samples= rand.sample(self.label_groups.get(label),self.k)
            for idx in K_samples:
                batch_list.append(idx)
        
        return batch_list


    def __iter__(self):

        for i in range(self.num_batches):
            yield self.get1batch()

    def __len__(self):

        return self.num_batches
            
