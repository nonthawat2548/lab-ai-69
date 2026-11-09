"""Restartable EfficientNet-B0 150-class training for Kaggle.
Input data: data/train/<class>/, data/val/<class>/, data/test/<class>/,
            data/class_names.json (ordered; 50 original classes first).
Never modifies the deployed 50-class checkpoint.
"""
import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def atomic_save(obj, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, filename)


def make_datasets(root, names):
    normal = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    aug = transforms.Compose([transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), normal])
    evaluation = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), normal])
    results = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(root / split, transform=aug if split == "train" else evaluation)
        if set(dataset.classes) != set(names):
            missing = set(names) - set(dataset.classes)
            extra = set(dataset.classes) - set(names)
            raise ValueError(f"{split} classes differ: missing={sorted(missing)}, extra={sorted(extra)}")
        relabel = {dataset.class_to_idx[name]: names.index(name) for name in dataset.classes}
        dataset.samples = [(file, relabel[label]) for file, label in dataset.samples]
        dataset.targets = [relabel[label] for label in dataset.targets]
        results[split] = dataset
    return results


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Prepared dataset folder")
    parser.add_argument("--old-checkpoint", required=True, help="Original 50-class .pth")
    parser.add_argument("--output-dir", default="/kaggle/working/thai_food_150")
    parser.add_argument("--resume", default="", help="Previous last_checkpoint.pth; must match classes")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs across resumes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid epochs, batch size, or workers")
    root = Path(args.data)
    names = json.loads((root / "class_names.json").read_text(encoding="utf-8"))
    if len(names) != 150 or len(set(names)) != 150:
        raise ValueError("Expected 150 distinct classes")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    datasets_by_split = make_datasets(root, names)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"),
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
        for split, dataset in datasets_by_split.items()
    }
    original = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)
    old_names = list(original["class_names"])
    if len(old_names) != 50 or names[:50] != old_names:
        raise ValueError("First 50 class names must match old checkpoint order exactly")
    old = models.efficientnet_b0(weights=None)
    old.classifier[1] = nn.Linear(old.classifier[1].in_features, 50)
    old.load_state_dict(original["model"], strict=True)
    network = models.efficientnet_b0(weights=None)
    network.classifier[1] = nn.Linear(network.classifier[1].in_features, 150)
    network.features.load_state_dict(old.features.state_dict())
    with torch.no_grad():
        network.classifier[1].weight[:50].copy_(old.classifier[1].weight)
        network.classifier[1].bias[:50].copy_(old.classifier[1].bias)
    del old, original
    network = network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_val = -1.0
    best_state = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["class_names"] != names:
            raise ValueError("Resume checkpoint classes differ from dataset")
        network.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scaler") and device.type == "cuda":
            scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"])
        best_val = float(saved["best_val"])
        best_state = saved["best_model"]
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)
    output = Path(args.output_dir).resolve()
    if output == Path(args.old_checkpoint).resolve().parent:
        raise ValueError("Output directory must not be legacy checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, args.epochs):
        for split in ("train", "val"):
            network.train(split == "train")
            correct = 0
            total = 0
            running_loss = 0.
            for batch, labels in loaders[split]:
                batch = batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        logits = network(batch)
                        loss = criterion(logits, labels)
                    if split == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.numel()
                running_loss += loss.detach().item() * labels.numel()
            accuracy = correct / max(total, 1)
            print(f"Epoch {epoch + 1}/{args.epochs} {split}: "
                  f"loss={running_loss/max(total,1):.4f} acc={accuracy:.4f}", flush=True)
            if split == "val" and accuracy > best_val:
                best_val = accuracy
                best_state = cpu_state(network)
        # Atomic save after EACH completed epoch; includes optimizer and best model.
        atomic_save({
            "epoch": epoch + 1, "class_names": names, "model": cpu_state(network),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_val": best_val, "best_model": best_state,
        }, output / "last_checkpoint.pth")
        atomic_save({"model": best_state, "class_names": names,
                     "validation_accuracy": best_val}, output / "food_model_150_best.pth")
        print("Checkpoint saved to", output, flush=True)
    if best_state is None:
        raise ValueError("No completed epochs. Increase --epochs above checkpoint epoch.")
    network.load_state_dict(best_state)
    network.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch, labels in loaders["test"]:
            predictions = network(batch.to(device)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    test_accuracy = correct / max(total, 1)
    atomic_save({"model": best_state, "class_names": names,
                 "validation_accuracy": best_val, "test_accuracy": test_accuracy},
                output / "food_model_150_best.pth")
    (output / "metrics.json").write_text(
        json.dumps({"epochs_completed": args.epochs,
                    "validation_accuracy": best_val,
                    "test_accuracy": test_accuracy,
                    "classes": len(names)}, indent=2), encoding="utf-8")
    print("Completed! Test accuracy:", round(test_accuracy, 4), flush=True)


if __name__ == "__main__":
    main()
