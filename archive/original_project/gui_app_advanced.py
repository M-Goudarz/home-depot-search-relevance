import os
import re
import pickle
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import messagebox

try:
    import customtkinter as ctk
except ImportError:
    os.system('pip install customtkinter')
    import customtkinter as ctk

from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer, util

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

FONT_MAIN = ("Vazirmatn", 13)
FONT_TITLE = ("Vazirmatn", 20, "bold")
FONT_LABEL = ("Vazirmatn", 12, "bold")
FONT_CODE = ("Consolas", 12)

model_regressor = None
tfidf_vectorizer = None
semantic_model = None
is_pipeline_loaded = False
stemmer = PorterStemmer()

home_depot_synonyms = {
    "galv": "galvanized", "gal": "gallon", "cu ft": "cubic feet",
    "ft": "foot", "in": "inch", "in.": "inch", "lbs": "pound",
    "lb": "pound", "w/": "with", "w/o": "without",
    "whirlpool": "whirlpool", "wirlpool": "whirlpool",
    "anderson": "andersen", "sheetrock": "drywall"
}

def clean_text_advanced(text):
    if isinstance(text, str):
        text = text.lower()
        
        words = text.split()
        words = [home_depot_synonyms[w] if w in home_depot_synonyms else w for w in words]
        text = " ".join(words)
        
        text = re.sub(r'(\d+)\s*(inches|inch|in|in\.)(?=\b|\d)', r'\1 inch ', text)
        text = re.sub(r'(\d+)\s*(feet|foot|ft|ft\.)(?=\b|\d)', r'\1 foot ', text)
        text = re.sub(r'(\d+)\s*(pounds|pound|lbs|lb|lb\.)(?=\b|\d)', r'\1 pound ', text)
        text = re.sub(r'(\d+)\s*(volts|volt|v)(?=\b|\d)', r'\1 volt ', text)
        text = re.sub(r'(\d+)\s*(watts|watt|w)(?=\b|\d)', r'\1 watt ', text)
        text = re.sub(r'(\d+)(x|\s*x\s*)(\d+)', r'\1 x \3', text)
        
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        return " ".join([stemmer.stem(w) for w in text.split()])
    return ""

def count_common_words(search, target):
    if isinstance(search, str) and isinstance(target, str):
        return sum(1 for word in search.split() if word in target.split())
    return 0

def get_jaccard_similarity(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        set1, set2 = set(str1.split()), set(str2.split())
        if not set1 or not set2: return 0.0
        return float(len(set1.intersection(set2))) / len(set1.union(set2))
    return 0.0

def extract_features(search_raw, title_raw, desc_raw, brand_raw):
    """تابع ماژولار و مستقل جهت مهندسی و استخراج ۱۳ ویژگی (حل ایراد نهم)"""
    clean_search = clean_text_advanced(search_raw)
    clean_title = clean_text_advanced(title_raw)
    clean_desc = clean_text_advanced(desc_raw)
    clean_brand = clean_text_advanced(brand_raw if brand_raw.strip() else "Unknown")

    search_len = len(clean_search.split())

    word_match_title = count_common_words(clean_search, clean_title)
    word_match_desc = count_common_words(clean_search, clean_desc)
    word_match_brand = count_common_words(clean_search, clean_brand)

    ratio_title = word_match_title / (search_len + 1e-5)
    ratio_desc = word_match_desc / (search_len + 1e-5)
    ratio_brand = word_match_brand / (search_len + 1e-5)

    jaccard_title = get_jaccard_similarity(clean_search, clean_title)
    jaccard_desc = get_jaccard_similarity(clean_search, clean_desc)
    jaccard_brand = get_jaccard_similarity(clean_search, clean_brand)

    search_tfidf = tfidf_vectorizer.transform([clean_search])
    title_tfidf = tfidf_vectorizer.transform([clean_title])
    desc_tfidf = tfidf_vectorizer.transform([clean_desc])

    tfidf_cos_title = float(np.sum(search_tfidf.multiply(title_tfidf)))
    tfidf_cos_desc = float(np.sum(search_tfidf.multiply(desc_tfidf)))

    search_emb = semantic_model.encode(clean_search, convert_to_tensor=True)
    title_emb = semantic_model.encode(clean_title, convert_to_tensor=True)
    bert_sim = float(util.cos_sim(search_emb, title_emb).item())

    features = [
        search_len, word_match_title, word_match_desc, word_match_brand,
        ratio_title, ratio_desc, ratio_brand,
        jaccard_title, jaccard_desc, jaccard_brand,
        tfidf_cos_title, tfidf_cos_desc, bert_sim
    ]
    
    return features, clean_search, clean_title, clean_brand, bert_sim

class AdvancedIRSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("سامانه ارزیابی هوشمند ارتباط جستجو - نسخه نهایی")
        self.geometry("1120x760")
        self.resizable(False, False)

        self.build_ui()
        threading.Thread(target=self.load_models_in_background, daemon=True).start()

    def build_ui(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="#1A1A24", corner_radius=0, height=85)
        self.header_frame.pack(fill="x", side="top")
        
        self.lbl_title = ctk.CTkLabel(
            self.header_frame, 
            text="موتور جستجو و ارزیابی معنایی محصولات (اطلاعات پیشرفته دیتاساینس)", 
            font=FONT_TITLE, text_color="#3B82F6"
        )
        self.lbl_title.pack(pady=22)

        self.tabview = ctk.CTkTabview(self, width=1040, height=590, segmented_button_selected_color="#2563EB")
        self.tabview.pack(padx=25, pady=10, fill="both", expand=True)
        
        self.tab_main = self.tabview.add("موتور ارزیابی هوشمند")
        self.tab_history = self.tabview.add("تاریخچه لاگ‌های سیستم")
        self.tab_about = self.tabview.add("ساختار فنی پروژه")

        self.build_main_tab()
        self.build_history_tab()
        self.build_about_tab()

        self.status_bar = ctk.CTkFrame(self, height=45, fg_color="#1A1A24", corner_radius=0)
        self.status_bar.pack(fill="x", side="bottom")
        
        self.lbl_status = ctk.CTkLabel(
            self.status_bar, 
            text="⏳ سیستم در حال بارگذاری وزن‌های BERT و LightGBM در هسته پس‌زمینه است...", 
            font=FONT_MAIN, text_color="#F59E0B"
        )
        self.lbl_status.pack(side="right", padx=25, pady=6)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_bar, width=220, mode="indeterminate")
        self.progress_bar.pack(side="left", padx=25, pady=12)
        self.progress_bar.start()

    def build_main_tab(self):
        card_frame = ctk.CTkFrame(self.tab_main, fg_color="#222230", corner_radius=16)
        card_frame.pack(fill="both", expand=True, padx=40, pady=25)

        self.lbl_search = ctk.CTkLabel(card_frame, text="عبارت جستجوی کاربر (Query):", font=FONT_LABEL)
        self.lbl_search.grid(row=0, column=1, padx=25, pady=(35, 8), sticky="e")
        self.entry_search = ctk.CTkEntry(card_frame, width=640, font=FONT_CODE, justify="left", fg_color="#111118")
        self.entry_search.grid(row=0, column=0, padx=25, pady=(35, 8), sticky="w")

        self.lbl_product = ctk.CTkLabel(card_frame, text="عنوان کالا در کاتالوگ (Title):", font=FONT_LABEL)
        self.lbl_product.grid(row=1, column=1, padx=25, pady=16, sticky="e")
        self.entry_title = ctk.CTkEntry(card_frame, width=640, font=FONT_CODE, justify="left", fg_color="#111118")
        self.entry_title.grid(row=1, column=0, padx=25, pady=16, sticky="w")

        self.lbl_brand = ctk.CTkLabel(card_frame, text="برند کارخانه (Brand - اختیاری):", font=FONT_LABEL)
        self.lbl_brand.grid(row=2, column=1, padx=25, pady=16, sticky="e")
        self.entry_brand = ctk.CTkEntry(card_frame, width=640, font=FONT_CODE, justify="left", fg_color="#111118")
        self.entry_brand.grid(row=2, column=0, padx=25, pady=16, sticky="w")

        self.lbl_desc = ctk.CTkLabel(card_frame, text="توضیحات مشخصات فنی:", font=FONT_LABEL)
        self.lbl_desc.grid(row=3, column=1, padx=25, pady=16, sticky="ne")
        self.text_desc = ctk.CTkTextbox(card_frame, width=640, height=90, font=FONT_CODE, fg_color="#111118", border_width=1, border_color="#2D2D3A")
        self.text_desc.grid(row=3, column=0, padx=25, pady=16, sticky="w")

        action_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        action_frame.grid(row=4, column=0, columnspan=2, pady=35, sticky="ew")

        self.btn_predict = ctk.CTkButton(
            action_frame, 
            text="⚡ محاسبه میزان ارتباط (Run Inference)", 
            font=("Vazirmatn", 14, "bold"),
            height=52, width=320,
            state="disabled", 
            command=self.run_prediction_logic
        )
        self.btn_predict.pack(side="left", padx=35)

        self.lbl_score_text = ctk.CTkLabel(action_frame, text="امتیاز ارتباط سیستم (۱ تا ۳):", font=FONT_LABEL)
        self.lbl_score_text.pack(side="right", padx=(0, 25))

        self.lbl_final_score = ctk.CTkLabel(action_frame, text="--", font=("Consolas", 38, "bold"), text_color="#A0A0AB")
        self.lbl_final_score.pack(side="right", padx=25)

    def build_history_tab(self):
        self.history_text = ctk.CTkTextbox(self.tab_history, width=940, height=460, font=FONT_CODE, fg_color="#111118", state="disabled")
        self.history_text.pack(pady=25)

    def build_about_tab(self):
        about_frame = ctk.CTkFrame(self.tab_about, fg_color="#222230", corner_radius=16)
        about_frame.pack(fill="both", expand=True, padx=40, pady=30)

        info_text = (
            "سیستم ارزیابی ترکیبی هوشمند (Hybrid Search Relevance Core)\n"
            "طراحی و پیاده‌سازی شده بر اساس آخرین استانداردهای پردازش زبان طبیعی (NLP)\n\n"
            "لایه‌های مهندسی داده اعمال شده در این محصول:\n"
            "۱. لایه واژگانی (Lexical Matcher): شمارش دقیق توکن‌های ریشه‌یابی شده مشترک با فرمول پورتر.\n"
            "۲. لایه اصلاح عبارات (Synonym Mapping & Typo Corrector): بازسازی مخفف‌های فنی قبل از پردازش آماری.\n"
            "۳. لایه برداری آماری (TF-IDF Vector Space): استخراج بردارهای فراوانی متن و محاسبه شباهت کسینوسی.\n"
            "۴. لایه معنایی عمیق (Dense Transformer Embeddings): استفاده از مدل پیش‌آموزش‌دیده BERT برای مترادف‌یابی مبل/کاناپه.\n"
            "۵. الگوریتم یادگیری نظارت‌شده (LightGBM Regressor): ترکیب ۱۳ ویژگی استخراج‌شده و تولید نمره با خطای کم.\n"
            "۶. لایه فیلتر ضد فریب (Keyword Penalty Layer): جریمه هوشمند نمرات فریبنده در سرچ‌های چندکلمه‌ای مفقود شده."
        )

        lbl_about = ctk.CTkLabel(about_frame, text=info_text, font=FONT_MAIN, justify="right")
        lbl_about.pack(padx=25, pady=35)

    def load_models_in_background(self):
        global model_regressor, tfidf_vectorizer, semantic_model, is_pipeline_loaded
        
        try:
            time.sleep(1.2)
            
            with open('ir_sota_search_model.pkl', 'rb') as f:
                saved_data = pickle.load(f)
            model_regressor = saved_data['trained_model']
            tfidf_vectorizer = saved_data['tfidf_vectorizer']
            
            semantic_model = SentenceTransformer('local_mini_bert_model')
            is_pipeline_loaded = True
            
            self.after(0, self.on_models_loaded_success)
            
        except Exception as e:
            self.after(0, lambda: self.on_models_loaded_error(str(e)))

    def on_models_loaded_success(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_status.configure(text="✅ تمامی فیلترها و وزن‌های هوش مصنوعی (BERT & LightGBM) آماده عملیات هستند.", text_color="#10B981")
        self.btn_predict.configure(state="normal")

    def on_models_loaded_error(self, error_msg):
        self.progress_bar.stop()
        self.lbl_status.configure(text="❌ لود مدل با خطا مواجه شد. وجود فایل pkl و پوشه برت را چک کنید.", text_color="#EF4444")
        messagebox.showerror("خطای بارگذاری خط لوله", f"خطای زیر رخ داده است:\n{error_msg}")

    def log_history(self, search, title, final_score, base_score, text_penalty):
        self.history_text.configure(state="normal")
        log_entry = (
            f"Query Term: '{search}'\n"
            f"Catalog Title: '{title}'\n"
            f"-> LightGBM Score: {base_score:.3f} | Penalty Applied: {text_penalty}\n"
            f"-> Adjusted Final Relevance Score: {final_score:.2f}\n"
            f"{'='*90}\n"
        )
        self.history_text.insert(tk.END, log_entry)
        self.history_text.configure(state="disabled")

    def run_prediction_logic(self):
        if not is_pipeline_loaded:
            return

        search_raw = self.entry_search.get()
        title_raw = self.entry_title.get()
        desc_raw = self.text_desc.get("1.0", tk.END)
        brand_raw = self.entry_brand.get()

        if not search_raw.strip() or not title_raw.strip():
            messagebox.showwarning("نقص در ورودی", "وارد کردن عبارت جستجو و عنوان محصول الزامی است.")
            return

        # فراخوانی تابع یکپارچه استخراج ویژگی (رفع تمیز ایراد نهم جی‌پتی)
        features, clean_search, clean_title, clean_brand, bert_sim = extract_features(
            search_raw, title_raw, desc_raw, brand_raw
        )

        prediction = model_regressor.predict([features])[0]
        base_prediction = prediction
        penalty_status = "No"

        search_words = clean_search.split()
        if len(search_words) > 1:
            missing_words = [w for w in search_words if w not in clean_title.split() and w not in clean_brand.split()]
            if len(missing_words) >= 1 and bert_sim < 0.75:
                prediction = prediction - (0.42 * len(missing_words))
                penalty_status = f"Yes (-{0.42 * len(missing_words):.2f})"

        prediction = max(1.0, min(3.0, prediction))

        self.lbl_final_score.configure(text=f"{prediction:.2f}")
        
        if prediction >= 2.30:
            self.lbl_final_score.configure(text_color="#10B981")
        elif prediction >= 1.70:
            self.lbl_final_score.configure(text_color="#F59E0B")
        else:
            self.lbl_final_score.configure(text_color="#EF4444")

        self.log_history(search_raw, title_raw, prediction, base_prediction, penalty_status)

if __name__ == "__main__":
    app = AdvancedIRSearchApp()
    app.mainloop()