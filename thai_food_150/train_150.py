"""Fine-tune the original 50-class EfficientNet-B0 into 150 classes."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data");p.add_argument("--old-checkpoint",default="food_model_best.pth")
    p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--output",default="food_model_150.pth")
    a=p.parse_args();torch.manual_seed(42);random.seed(42);np.random.seed(42)
    root=Path(a.data);names=json.loads((root/"class_names.json").read_text(encoding="utf-8"))
    if len(names)!=150:raise ValueError("Expected 150 classes")
    normal=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normal])
    eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normal])
    dsets={s:datasets.ImageFolder(root/s,transform=train_tf if s=="train" else eval_tf) for s in ("train","val","test")}
    for d in dsets.values():
        if set(d.classes)!=set(names):raise ValueError("Missing classes in one split")
        mapping={d.class_to_idx[name]:names.index(name) for name in d.classes}
        d.samples=[(f,mapping[y]) for f,y in d.samples]
        d.targets=[mapping[y] for y in d.targets]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=s=="train",num_workers=2) for s,d in dsets.items()}
    checkpoint=torch.load(a.old_checkpoint,map_location="cpu",weights_only=False)
    old_names=list(checkpoint["class_names"])
    old=models.efficientnet_b0(weights=None);old.classifier[1]=nn.Linear(old.classifier[1].in_features,len(old_names))
    old.load_state_dict(checkpoint["model"])
    model=models.efficientnet_b0(weights=None);model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(names))
    model.features.load_state_dict(old.features.state_dict())
    with torch.no_grad():
        for i,name in enumerate(old_names):
            j=names.index(name);model.classifier[1].weight[j].copy_(old.classifier[1].weight[i])
            model.classifier[1].bias[j].copy_(old.classifier[1].bias[i])
    model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0002);loss_fn=nn.CrossEntropyLoss()
    best=-1;best_state=None
    for epoch in range(a.epochs):
        for split in ("train","val"):
            model.train(split=="train");correct=total=0
            for x,y in loaders[split]:
                x=x.to(device);y=y.to(device)
                with torch.set_grad_enabled(split=="train"):
                    logits=model(x);loss=loss_fn(logits,y)
                    if split=="train":opt.zero_grad();loss.backward();opt.step()
                correct+=(logits.argmax(1)==y).sum().item();total+=y.numel()
            acc=correct/max(total,1);print(epoch+1,split,round(acc,4),flush=True)
            if split=="val" and acc>best:
                best=acc;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    if best_state is None:raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state);model.eval();correct=total=0
    with torch.no_grad():
        for x,y in loaders["test"]:
            out=model(x.to(device)).argmax(1).cpu()
            correct+=(out==y).sum().item();total+=len(y)
    test_acc=correct/max(total,1)
    torch.save({"model":best_state,"class_names":names,"validation_accuracy":best,"test_accuracy":test_acc},a.output)
    print("Saved",a.output,"val",best,"test",test_acc)
if __name__=="__main__":main()
