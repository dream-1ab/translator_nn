
from pathlib import Path
import torch
from model import MyTranslator
from torch.nn import CrossEntropyLoss

def save_model(model: MyTranslator, optimizer: torch.optim.Adam, grad_scaler: torch.GradScaler, criterion: CrossEntropyLoss, epoch: int, counter: int):
    import os
    dir = Path(__file__).parent.parent.parent / "checkpoint"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterion": criterion.state_dict(),
        "epoch": epoch,
        "counter": counter,
        "grad_scaler": grad_scaler.state_dict()
    },  dir / "checkpoint.pth")
    
    torch.save(model.state_dict(), dir / "my_translator.pth")


def load_model(model: MyTranslator, optimizer: torch.optim.Adam, grad_scaler: torch.cuda.amp.GradScaler, criterion: CrossEntropyLoss, checkpoint_path: Path = Path(__file__).parent.parent.parent / "checkpoint" / "checkpoint.pth") -> tuple[int, int]:
    path = Path(checkpoint_path)
    if not path.exists():
        return 0, 0
    checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer"])
    grad_scaler.load_state_dict(checkpoint.get("grad_scaler", {}))  # safe fallback
    criterion.load_state_dict(checkpoint["criterion"])
    
    epoch: int = checkpoint.get("epoch", 0)
    counter: int = checkpoint.get("counter", 0)
    return epoch, counter
