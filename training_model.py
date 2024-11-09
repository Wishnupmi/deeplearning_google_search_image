import os
import numpy as np
import torch
import clip
from PIL import Image
import pickle
from concurrent.futures import ThreadPoolExecutor

# Inisialisasi model CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Fungsi untuk ekstraksi embedding gambar
def extract_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image)
    return image_embedding.cpu().numpy()

# Fungsi untuk ekstraksi embedding teks
def extract_text_features(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding.cpu().numpy()

# Memuat gambar dan deskripsi teks dari folder
def load_data(image_folder):
    image_embeddings = []
    text_embeddings = []
    labels = []
    
    # List untuk memuat tugas-tugas gambar yang akan diproses
    tasks = []
    
    # Fungsi untuk memproses gambar dan deskripsi teks
    def process_image(label, img_file):
        img_path = os.path.join(image_folder, label, img_file)
        image_embedding = extract_image_features(img_path)
        text_embedding = extract_text_features(f"Seekor {label}")  # Deskripsi teks
        
        return image_embedding, text_embedding, label
    
    # Menyiapkan ThreadPoolExecutor untuk pemrosesan paralel
    with ThreadPoolExecutor() as executor:
        for label in os.listdir(image_folder):
            label_folder = os.path.join(image_folder, label)
            if os.path.isdir(label_folder):
                for img_file in os.listdir(label_folder):
                    tasks.append(executor.submit(process_image, label, img_file))
        
        # Mengambil hasil dari setiap tugas
        for future in tasks:
            image_embedding, text_embedding, label = future.result()
            image_embeddings.append(image_embedding)
            text_embeddings.append(text_embedding)
            labels.append(label)
    
    return np.array(image_embeddings), np.array(text_embeddings), labels

# Folder berisi gambar kucing dan anjing
image_folder = "data/gambar"
image_embeddings, text_embeddings, labels = load_data(image_folder)

# Simpan embedding yang telah dilatih
with open('trained_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'image_embeddings': image_embeddings,
        'text_embeddings': text_embeddings,
        'labels': labels
    }, f)

print("Training selesai, embedding gambar dan teks sudah disimpan.")
