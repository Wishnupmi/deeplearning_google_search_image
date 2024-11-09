import os
import numpy as np
import torch
import clip
from PIL import Image
import pickle  # Make sure pickle is imported
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # Importing Matplotlib
import random  # Importing random untuk acak

# Memuat model CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Fungsi untuk ekstraksi embedding teks
def extract_text_features(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding.cpu().numpy()

# Memuat embedding gambar yang telah dilatih
with open('trained_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    image_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
    labels = data['labels']

# Fungsi untuk menghitung kesamaan kosinus
def calculate_cosine_similarity(image_embedding, text_embedding):
    return cosine_similarity(image_embedding, text_embedding)[0][0]

# Fungsi untuk menampilkan gambar berdasarkan kesamaan dengan teks secara acak
def display_image_for_text(query_text, image_folder, image_embeddings, labels, top_k=5):
    # Ekstraksi embedding untuk teks (kata yang dicari)
    text_embedding = extract_text_features(query_text)

    # Menghitung kesamaan antara teks dan gambar
    similarities = []
    for img_embedding, label in zip(image_embeddings, labels):
        similarity = calculate_cosine_similarity(img_embedding, text_embedding)
        similarities.append((similarity, label))
    
    # Urutkan berdasarkan kemiripan tertinggi
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Menampilkan gambar sesuai dengan top_k secara acak
    count = 0
    for similarity, label in similarities:
        # Mencari gambar dengan label yang sesuai
        label_folder = os.path.join(image_folder, label)
        image_files = os.listdir(label_folder)
        
        # Acak urutan gambar yang ditemukan di dalam folder
        random.shuffle(image_files)  # Acak urutan gambar dalam folder label
        
        # Menampilkan gambar untuk setiap file yang ditemukan dalam folder label
        for img_file in image_files:
            img_path = os.path.join(label_folder, img_file)
            img = Image.open(img_path)
            
            # Menampilkan gambar dengan Matplotlib
            plt.imshow(img)
            plt.axis('off')  # Menonaktifkan axis
            plt.show()  # Menampilkan gambar
            
            print(f"Gambar yang ditampilkan: {img_path} dengan kemiripan {similarity:.4f}")
            
            # Meningkatkan counter dan mengecek apakah kita sudah menampilkan top_k gambar
            count += 1
            if count >= top_k:
                return  # Berhenti setelah menampilkan top_k gambar

# Menguji dengan teks "kucing", menampilkan 100 gambar secara acak
display_image_for_text("kucing", "data/gambar", image_embeddings, labels, top_k=100)
