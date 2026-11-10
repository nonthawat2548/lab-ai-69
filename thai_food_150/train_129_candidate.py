"""Fine-tune 120-class ThaiFood checkpoint with nine extra THDESS classes.
Never deploy automatically. Resumes from Drive checkpoints saved after each epoch.
"""
import argparse,copy,json,os,random,shutil,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms

def atomic_torch_save(payload,path):
    path=Path(path)
    local=Path("/content")/(path.name+".pending")
    torch.save(payload,local)
    uploading=path.with_name(path.name+".uploading")
    shutil.copy2(local,uploading)
    os.replace(uploading,path)
    local.unlink(missing_ok=True)

def load_and_extract(zips,folder):
    marker=folder/".completed"
    if marker.exists():return
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for filename in zips:
        print("Extracting",filename.name,flush=True)
        with zipfile.ZipFile(filename) as archive:
            for m in archive.infolist():
                if m.is_dir():continue
                dest=(folder/m.filename).resolve()
                if not dest.is_relative_to(folder.resolve()):raise ValueError("Unsafe ZIP path")
                if dest.exists():raise ValueError("ZIP filename collision: "+str(dest))
            archive.extractall(folder)
    marker.write_text("complete")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="/content/thaifood_drive/MyDrive/ThaiFood150")
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=16)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:raise ValueError("Invalid training settings")
    base=Path(args.base);prepared=base/"prepared";out=base/"training_129"
    out.mkdir(parents=True,exist_ok=True)
    original=base/"training_120/best_model_120.pth"
    names129=json.loads((prepared/"class_names_129.json").read_text(encoding="utf-8"))
    names120=json.loads((prepared/"class_names_120.json").read_text(encoding="utf-8"))
    if len(names129)!=129 or len(set(names129))!=129 or names129[:120]!=names120:
        raise ValueError("129-class mapping is invalid or old 120 order changed")
    zips=[prepared/"ThaiFood120_merged.zip",prepared/"ThaiFood129_extra_desserts.zip"]
    for f in zips+[original]:
        if not f.exists():raise FileNotFoundError(str(f))
    if not all(zipfile.is_zipfile(z) for z in zips):raise ValueError("One dataset ZIP invalid")
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":raise RuntimeError("Select GPU accelerator in Colab")
    print("Training on",torch.cuda.get_device_name(0),flush=True)
    folder=Path("/content/thaifood129_data")
    load_and_extract(zips,folder)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
    eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
    loaders={}
    for split in ("train","val","test"):
        ds=datasets.ImageFolder(folder/split,transform=train_tf if split=="train" else eval_tf)
        if set(ds.classes)!=set(names129):raise ValueError(f"Missing classes in {split}")
        remap={ds.class_to_idx[name]:names129.index(name) for name in ds.classes}
        ds.samples=[(f,remap[y]) for f,y in ds.samples]
        ds.targets=[y for _,y in ds.samples]
        loaders[split]=DataLoader(ds,batch_size=args.batch_size,
             shuffle=(split=="train"),num_workers=2,pin_memory=True)
        print(split,len(ds),flush=True)
    oldckpt=torch.load(original,map_location="cpu",weights_only=False)
    if list(oldckpt["class_names"])!=names120:raise ValueError("120 checkpoint label order differs")
    old=models.efficientnet_b0(weights=None)
    old.classifier[1]=nn.Linear(old.classifier[1].in_features,120)
    old.load_state_dict(oldckpt["model"],strict=True)
    model=models.efficientnet_b0(weights=None)
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,129)
    model.features.load_state_dict(old.features.state_dict())
    with torch.no_grad():
        model.classifier[1].weight[:120].copy_(old.classifier[1].weight)
        model.classifier[1].bias[:120].copy_(old.classifier[1].bias)
    del old,oldckpt
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.00015)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    last=out/"last_checkpoint.pth";best=out/"best_model_129.pth"
    start=0;best_acc=-1.
    if last.is_file():
        ck=torch.load(last,map_location="cpu",weights_only=False)
        if ck["class_names"]!=names129:raise ValueError("Resume classes mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        scaler.load_state_dict(ck["scaler"])
        start=ck["epoch"];best_acc=ck["best_accuracy"]
        print("Resume from epoch",start+1,flush=True)
    for epoch in range(start,args.epochs):
        metrics={}
        for split in ("train","val"):
            training=split=="train";model.train(training)
            correct=total=0;running=0.
            for x,y in loaders[split]:
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                if training:optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type="cuda",enabled=True):
                        output=model(x);loss=criterion(output,y)
                    if training:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer);scaler.update()
                total+=len(y);correct+=(output.argmax(1)==y).sum().item()
                running+=loss.item()*len(y)
            metrics[split]={"accuracy":correct/max(total,1),"loss":running/max(total,1)}
            print(f"epoch {epoch+1}/{args.epochs} {split}: {metrics[split]}",flush=True)
        val=metrics["val"]["accuracy"]
        if val>best_acc:
            best_acc=val
            atomic_torch_save({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
              "class_names":names129,"validation_accuracy":best_acc,"epoch":epoch+1},best)
        atomic_torch_save({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
          "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
          "class_names":names129,"epoch":epoch+1,"best_accuracy":best_acc},last)
        history=out/"history.jsonl"
        with history.open("a",encoding="utf-8") as h:
            h.write(json.dumps({"epoch":epoch+1,**metrics})+"\n")
        print("Saved checkpoint",epoch+1,flush=True)
    if start>=args.epochs:print("All requested epochs already complete")
    best_weights=torch.load(best,map_location="cpu",weights_only=False)
    model.load_state_dict(best_weights["model"])
    model.eval();correct=total=0
    with torch.inference_mode():
        for x,y in loaders["test"]:
            pred=model(x.to(device)).argmax(1).cpu()
            correct+=(pred==y).sum().item();total+=len(y)
    test_accuracy=correct/max(total,1)
    (out/"results.json").write_text(json.dumps({"classes":129,"best_val_accuracy":best_acc,
        "test_accuracy":test_accuracy,"test_images":total,"status":"candidate_not_deployed"},indent=2))
    print("Finished 129-class candidate. Test accuracy:",test_accuracy,flush=True)
    print("Still need 21 genuinely distinct, image-backed classes for 150.",flush=True)

if __name__=="__main__":main()
