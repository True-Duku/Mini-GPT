import torch
import torch.nn as nn
import numpy as np
from transformers import FlaubertTokenizer
from tqdm import tqdm
#import os

# === SETUP ===
tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
vocab_size = tokenizer.vocab_size

# === Détection automatique du GPU ou CPU
def safe_device():
    try:
        torch.cuda.empty_cache()
        torch.tensor([0.0]).cuda()
        print("Entraînement sur GPU")
        return torch.device("cuda")
    except:
        print("GPU indisponible ou mémoire insuffisante — bascule sur CPU")
        return torch.device("cpu")

device = safe_device()

# === MiniGPT ===
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=50, num_heads=16, ff_dim=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.positional_enc = self.create_positional_encoding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                pe[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_emb(input_ids) + self.positional_enc[:, :seq_len, :].to(input_ids.device)
        x = x.transpose(0, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        memory = torch.zeros(1, input_ids.size(0), x.size(-1), device=input_ids.device)
        decoded = self.decoder(x, memory, tgt_mask=mask)
        return self.out_proj(decoded.transpose(0, 1))

# === Chargement du corpus en blocs ===
def stream_corpus_in_blocks(file_path, block_size):
    with open(file_path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            block.append(line)
            if len(block) >= block_size:
                yield block
                block = []
        if block:
            yield block

# === Fonction d'entraînement ===
def train(model, inputs, targets, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, target_ids in zip(inputs, targets):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Époque {epoch + 1} terminée — Perte totale : {total_loss:.4f}")

# === Génération de texte ===
def generate(model, start_text, max_len=20):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
    generated = input_ids.tolist()[0]
    for _ in range(max_len):
        input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.softmax(output[0, -1], dim=0)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated)

# === Initialisation du modèle et optimisation ===
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# === Paramètres de corpus ===
corpus_path = "Corpus.txt"  # à adapter
block_size = 5000

# === Boucle principale d'entraînement avec sauvegarde et bascule CPU/GPU ===
for block_num, lines in enumerate(stream_corpus_in_blocks(corpus_path, block_size)):
    print(f"\n Bloc {block_num + 1} — {len(lines)} phrases")
    inputs, targets = [], []

    for line in tqdm(lines, desc=f"Préparation du bloc {block_num + 1}"):
        tokenized = tokenizer(line, return_tensors="pt", truncation=True, max_length=50)["input_ids"]
        target = tokenized.clone()
        target[:, :-1] = tokenized[:, 1:]
        inputs.append(tokenized.to(device))
        targets.append(target.to(device))

    try:
        train(model, inputs, targets, optimizer, criterion)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Mémoire GPU saturée — bascule sur CPU pour ce bloc")
            device = torch.device("cpu")
            model.to(device)
            inputs = [i.cpu() for i in inputs]
            targets = [t.cpu() for t in targets]
            train(model, inputs, targets, optimizer, criterion)
        else:
            raise e

    # Sauvegarde du modèle après chaque bloc
    #checkpoint_name = f"minigpt_bloc_{block_num + 1:02d}.pth"
    #torch.save(model.state_dict(), checkpoint_name)
    #print(f" Modèle sauvegardé : {checkpoint_name}")

# === SAUVEGARDE FINALE UNIQUE DU MODÈLE ===
# Ce bloc de code est en dehors de la boucle,
# donc il est exécuté une seule fois à la fin.
print("\nSauvegarde finale du modèle...")
model_path = "mini_gpt_model_final.pth"
torch.save(model.state_dict(), model_path)
print(f"Modèle final sauvegardé dans {model_path}")

# === Génération interactive ===
user_input = input("Pose-moi une question : ")
response = generate(model, user_input)
print("Texte généré :", response)