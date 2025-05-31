
from torch.utils.data import Dataset, DataLoader
from dataset import SentencePairDataset, collete_fn, TranslationDirection
from tqdm import tqdm
from model import MyTranslator
from tokenizers import Tokenizer, decoders
import torch
from torch.nn import CrossEntropyLoss
from torch import Tensor
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tools.model_loader import load_model, save_model

def eval(model: MyTranslator, tokenizer: Tokenizer, data_loader: DataLoader, device: str) -> float:
    model.eval()
    criterion = CrossEntropyLoss(ignore_index=0)
    all_loss = 0.0
    train_progress = data_loader
    for (source_sentence, target_sentence) in train_progress:
        source_sentence: Tensor; target_sentence: Tensor
        source_sentence = source_sentence.to(device)
        target_sentence = target_sentence.to(device)

        target_input = target_sentence[:, :-1]
        target_output = target_sentence[:, 1:]

        with torch.no_grad():
            memory, memory_mask = model.generate_memory(x=source_sentence)
            output: Tensor = model(x=target_input, memory=memory, memory_key_padding_mask=memory_mask)
        output = output.reshape(-1, output.size(-1))
        target_output = target_output.reshape(-1)
        loss: Tensor = criterion(output, target_output)
        all_loss += loss.item()
    return all_loss

def train(model: MyTranslator, tokenizer: Tokenizer, trainset_loader: DataLoader, validationset_loader: DataLoader, n_epoch: int, device: str, summary_writer: SummaryWriter, learning_rate: 0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss(ignore_index=0)
    grad_scaler = torch.GradScaler()
    
    epoch_start, counter_start = load_model(model=model, optimizer=optimizer, grad_scaler=grad_scaler, criterion=criterion)
    
    counter = counter_start
    for i in range(epoch_start, n_epoch):
        train_progress = tqdm(trainset_loader, ncols=200)
        for (source_sentence, target_sentence) in train_progress:
            source_sentence: Tensor; target_sentence: Tensor
            source_sentence = source_sentence.to(device)
            target_sentence = target_sentence.to(device)

            target_input = target_sentence[:, :-1]
            target_output = target_sentence[:, 1:]
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                memory, memory_mask = model.generate_memory(x=source_sentence)
                output: Tensor = model(x=target_input, memory=memory, memory_key_padding_mask=memory_mask)
                # Zero gradients
                # Compute loss
                output = output.reshape(-1, output.size(-1))
                target_output = target_output.reshape(-1)
                loss: Tensor = criterion(output, target_output)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if counter % 200 == 0:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    #eval model performance using trainset.
                    validation_loss: float = eval(model, tokenizer, validationset_loader, device)
                    train_progress.set_description(f"epoch: {i}, train_loss: {str(loss.item())[:6]}, val_loss: {str(validation_loss)[:6]}")
                    #add tensorboard logging.
                    summary_writer.add_scalar("train loss", loss.item(), counter)
                    summary_writer.add_scalar("val loss", validation_loss, counter)
                model.train()
            counter += 1
        
        save_model(model=model, optimizer=optimizer, grad_scaler=grad_scaler, criterion=criterion, epoch=i, counter=counter)




# device = "mps"
device = "cuda:0"

tokenizer: Tokenizer = Tokenizer.from_file((Path(__file__).parent.parent / "config" / "./multilingual_tokenizer.json").as_posix())
tokenizer.decoder = decoders.Metaspace()
model = MyTranslator(d_model=512, n_vocab=tokenizer.get_vocab_size(), n_head=8, n_layer=6).to(device)
writer = SummaryWriter(Path(__file__).parent.parent / ".logs")

data_dir = Path(__file__).parent.parent / ".data"
tsvs = [
    f for f in data_dir.iterdir() if f.is_file() and f.name.endswith(".tsv")
]

directions: list[TranslationDirection] = [
    {"source": "ug", "target": "zh"},
    {"source": "zh", "target": "ug"},
    {"source": "ug", "target": "en"},
    {"source": "en", "target": "ug"},
]

train_set, validation_set = SentencePairDataset(tokenizer=tokenizer, tsv_files=[{"path": tsv, "directions": directions} for tsv in tsvs], max_allowed_tokens=300).split_into_train_set_validation_set(trainset_count=4096)
trainset_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collete_fn)
validationset_loader = DataLoader(validation_set, batch_size=64, shuffle=False, collate_fn=collete_fn)

train(model=model, tokenizer=tokenizer, trainset_loader=trainset_loader, validationset_loader=validationset_loader, n_epoch=200, device=device, summary_writer=writer, learning_rate=0.00005)
