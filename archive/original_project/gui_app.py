import tkinter as tk
from tkinter import messagebox
import pickle
import re
from nltk.stem.porter import PorterStemmer

# 1. Initialize Stemmer and Load the pre-trained model
stemmer = PorterStemmer()

try:
    with open('ir_search_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data['trained_model']
    feature_cols = saved_data['features']
except FileNotFoundError:
    print("Error: 'ir_search_model.pkl' not found. Please run the notebook cell first.")
    exit()

# 2. Re-define the exact cleaning and feature extraction functions for the GUI
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'(\d+)\s*(inches|inch|in|in\.)(?=\b|\d)', r'\1 inch ', text)
        text = re.sub(r'(\d+)\s*(feet|foot|ft|ft\.)(?=\b|\d)', r'\1 foot ', text)
        text = re.sub(r'(\d+)\s*(pounds|pound|lbs|lb|lb\.)(?=\b|\d)', r'\1 pound ', text)
        text = re.sub(r'(\d+)\s*(volts|volt|v)(?=\b|\d)', r'\1 volt ', text)
        text = re.sub(r'(\d+)(x|\s*x\s*)(\d+)', r'\1 x \3', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = text.split()
        return " ".join([stemmer.stem(w) for w in words])
    return ""

def count_common_words(search, target):
    if isinstance(search, str) and isinstance(target, str):
        search_words = search.split()
        target_words = target.split()
        return sum(1 for word in search_words if word in target_words)
    return 0

def get_jaccard_similarity(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        set1 = set(str1.split())
        set2 = set(str2.split())
        if not set1 or not set2: return 0.0
        return float(len(set1.intersection(set2))) / len(set1.union(set2))
    return 0.0

# 3. Predict Function triggered by the GUI Button
def calculate_relevance():
    # Get text from UI fields
    search_raw = entry_search.get()
    title_raw = entry_title.get()
    desc_raw = entry_desc.get("1.0", tk.END)
    
    if not search_raw.strip() or not title_raw.strip():
        messagebox.showwarning("Input Error", "Please fill in at least the Search Term and Product Title.")
        return
    
    # Preprocess inputs
    clean_search = clean_text(search_raw)
    clean_title = clean_text(title_raw)
    clean_desc = clean_text(desc_raw)
    
    # Extract features exactly like training
    search_len = len(clean_search.split())
    word_match_title = count_common_words(clean_search, clean_title)
    word_match_desc = count_common_words(clean_search, clean_desc)
    
    ratio_title = word_match_title / (search_len + 1e-5)
    ratio_desc = word_match_desc / (search_len + 1e-5)
    
    jaccard_title = get_jaccard_similarity(clean_search, clean_title)
    jaccard_desc = get_jaccard_similarity(clean_search, clean_desc)
    
    # Arrange features in a 2D array for the model
    input_features = [[search_len, word_match_title, word_match_desc, 
                       ratio_title, ratio_desc, jaccard_title, jaccard_desc]]
    
    # Predict using Random Forest
    prediction = model.predict(input_features)[0]
    
    # Clip prediction between 1.0 and 3.0 to keep it within real bounds
    prediction = max(1.0, min(3.0, prediction))
    
    # Display the result
    label_result.config(text=f"Predicted Relevance Score: {prediction:.2f} / 3.00", fg="green")

# 4. Building the Tkinter Window
root = tk.Tk()
root.title("Home Depot Search Relevance Predictor")
root.geometry("550x450")

# UI Labels and Inputs
tk.Label(root, text="Home Depot IR Search Project", font=("Arial", 16, "bold")).pack(pady=10)

tk.Label(root, text="Enter Search Term:", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
entry_search = tk.Entry(root, width=60)
entry_search.pack(pady=5, padx=20)

tk.Label(root, text="Enter Product Title:", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
entry_title = tk.Entry(root, width=60)
entry_title.pack(pady=5, padx=20)

tk.Label(root, text="Enter Product Description:", font=("Arial", 10, "bold")).pack(anchor="w", padx=20)
entry_desc = tk.Text(root, width=60, height=5)
entry_desc.pack(pady=5, padx=20)

# Calculate Button
btn_calculate = tk.Button(root, text="Predict Relevance Score", font=("Arial", 11, "bold"), 
                          bg="#4CAF50", fg="white", command=calculate_relevance)
btn_calculate.pack(pady=15)

# Result Label
label_result = tk.Label(root, text="Predicted Relevance Score: --", font=("Arial", 12, "bold"))
label_result.pack(pady=10)

root.mainloop()