


import torch
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, Module, Linear, CrossEntropyLoss, Dropout, ReLU
import math
from tokenizers import Tokenizer, decoders
from typing import Callable

class PositionalEncoder(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class EncoderDecoderLinearTransformer(Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dropper = Dropout(0.3)
        self.activator = ReLU()
        self.layer0 = Linear(in_features=d_model, out_features=d_model)
        self.layer1 = Linear(in_features=d_model, out_features=d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.dropper(x)
        x = self.activator(x)
        x = self.layer1(x)
        return x

class MyTranslator(torch.nn.Module):
    def __init__(self, d_model: int, n_vocab: int, n_head: int, n_layer: int):
        super().__init__()
        self.d_model = d_model
        self.shared_embedding = torch.nn.Embedding(n_vocab, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_len=4096)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=0.15, activation="gelu", batch_first=True,)
        self.encoder = TransformerEncoder(self.encoder_layer, n_layer)
        # self.encoder_decoder_linear_transformer = EncoderDecoderLinearTransformer(d_model=d_model)
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dropout=0.15, activation="gelu", batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, n_layer)
        self.vocab_classifier = Linear(in_features=d_model, out_features=n_vocab)
    
    def generate_memory(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        returns tuple of memory and memory key padding mask.
        """
        mask = x == 0
        # Pass input through embedding and positional encoder, then through encoder to get memory.
        # x: [batch_size, seq_len]
        embedded: Tensor = self.shared_embedding(x) * math.sqrt(self.d_model)
        embedded = self.positional_encoder(embedded)
        memory: Tensor = self.encoder(embedded, src_key_padding_mask=mask)
        # memory = self.encoder_decoder_linear_transformer(memory)
        return memory, mask
    
    def forward(self, x: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        """
        Forward pass for the decoder.
        Args:
            x: Tensor, shape [batch_size, tgt_seq_len]
            memory: Tensor, shape [batch_size, src_seq_len, d_model]
            memory_key_padding_mask: Tensor, shape [batch_size, src_seq_len]
        Returns:
            logits: Tensor, shape [batch_size, tgt_seq_len, n_vocab]
        """
        tgt_key_padding_mask = x == 0  # padding mask for target
        embedded = self.shared_embedding(x) * math.sqrt(self.d_model)
        embedded = self.positional_encoder(embedded)
        # Generate subsequent mask for causal decoding
        seq_len = x.size(1)
        device = x.device
        subsequent_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # Pass through decoder
        decoded = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=subsequent_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.vocab_classifier(decoded)
        return logits

    @torch.no_grad()
    def generate_text(self, prompt: str, tokenizer: Tokenizer, device: str, max_length: int = 256, on_token: Callable[[str], None] | None = None) -> str:
        self.eval()
        """
        Generate text from a prompt string using greedy decoding.
        Args:
            prompt: str, input sentence (in English)
            tokenizer: Tokenizer, tokenizer object
            max_length: int, maximum length of generated sequence
        Returns:
            str: generated Uyghur sentence
        """

        # Encode the prompt (English) and prepare as input
        prompt_ids = tokenizer.encode(f"<SOS><zh>{prompt}</zh><EOS>").ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # [1, src_seq_len]

        # Generate encoder memory from the prompt
        memory, memory_key_padding_mask = self.generate_memory(prompt_tensor)

        # Get special token ids
        sos_token_id: int = tokenizer.token_to_id("<SOS>")
        eos_token_id: int = tokenizer.token_to_id("<EOS>")

        # Start with <SOS>
        generated = torch.tensor([tokenizer.encode("<SOS><zh>").ids], dtype=torch.long, device=device)
        for i in range(max_length - 1):
            # print(f"--------{i}---------")
            logits = self(generated, memory, memory_key_padding_mask)  # [1, seq_len, vocab]
            # print("--------1---------")
            next_token_logits = logits[:, -1, :]  # [1, vocab]
            next_token = torch.argmax(next_token_logits, dim=-1)  # [1]
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == eos_token_id:
                break
            if on_token != None:
                on_token(tokenizer.decode([0, next_token.item(), 0], skip_special_tokens=True)[5:-5])
        # print("---------generated--------")
        # Remove <SOS> and everything after <EOS>
        output_ids = generated[0].tolist()
        # Decode to string
        output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        return output_text
