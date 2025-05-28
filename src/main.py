from torch.utils.data import Dataset, DataLoader
from dataset import SentencePairDataset, collete_fn
from tqdm import tqdm
from model import MyTranslator
from tokenizers import Tokenizer, decoders
import torch
from torch.nn import CrossEntropyLoss
from torch import Tensor
from pathlib import Path
from model import MyTranslator

# device = "mps"
device = "cuda:1"

tokenizer: Tokenizer = Tokenizer.from_file((Path(__file__).parent.parent / "config" / "./multilingual_tokenizer.json").as_posix())
tokenizer.decoder = decoders.Metaspace()
model = MyTranslator(d_model=512, n_vocab=tokenizer.get_vocab_size(), n_head=8, n_layer=6).to(device)

def load_model(model: MyTranslator):
    model.load_state_dict(torch.load(Path(__file__).parent.parent / "checkpoint" / "my_translator.pth"))

load_model(model)

while True:
    prompt = input("/>")
    def on_token(token: str):
        print(token, end="", flush=True)
    result = model.generate_text(prompt, tokenizer, device, 1024)
    print(f"{result}\n")

