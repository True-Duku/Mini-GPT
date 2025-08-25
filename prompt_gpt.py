import torch
import torch.nn as nn
from transformers import FlaubertTokenizer
import numpy as np
import os

# --- PARTIE 1 : CLASSES ET UTILITAIRES ---
# Copiez ici les classes et fonctions de votre script d'entraînement
# MiniGPT, safe_device, generate...

# === Détection automatique du GPU ou CPU
def safe_device():
    try:
        torch.cuda.empty_cache()
        torch.tensor([0.0]).cuda()
        print("Utilisation du GPU")
        return torch.device("cuda")
    except:
        print("GPU indisponible — bascule sur CPU")
        return torch.device("cpu")

device = safe_device()

# === MiniGPT ===
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_len=50, num_heads=16, ff_dim=512):
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

# --- PARTIE 2 : CHARGEMENT DU MODÈLE ET EXÉCUTION ---
if __name__ == "__main__":
    # --- Configuration ---
    # Remplacez ceci par votre tokenizer si vous n'utilisez pas Flaubert
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    vocab_size = tokenizer.vocab_size
    model_path = "mini_gpt_model_final.pth"

    # --- Chargement du modèle ---
    print("Chargement du modèle...")
    model = MiniGPT(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modèle chargé avec succès.")

    # --- Génération de texte ---
    user_input = input("Entrez votre prompt (q pour quitter) : ")
    while user_input.lower() != 'q':
        response = generate(model, user_input)
        print("Réponse générée :", response)
        user_input = input("Entrez un autre prompt (q pour quitter) : ")