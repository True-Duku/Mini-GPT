from tokenizers import ByteLevelBPETokenizer
#from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer # Pour charger le tokenizer enregistré

# Utilise ton fichier texte directement
# Assure-toi que ce fichier est au bon emplacement
files = ["sentences.txt"]

# Initialiser le tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.pre_tokenizer = Whitespace()

print("Début de l'entraînement du tokenizer sur le corpus...")
tokenizer.train(
    files,
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

print("Entraînement terminé.")

tokenizer.save("my_tokenizer.json")
print("Tokenizer basé sur le corpus sauvegardé sous 'my_tokenizer.json'")

# Charger et tester
loaded_tokenizer = Tokenizer.from_file("my_tokenizer.json")
test_text = "Un exemple de texte pour tester le tokenizer."

print(f"\nTokenisation d'un texte de test : {loaded_tokenizer.encode(test_text).tokens}")