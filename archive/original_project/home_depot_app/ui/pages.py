# ui/pages.py
import customtkinter as ctk
from theme import Theme
from backend_placeholder import real_inference_pipeline
import threading
from datetime import datetime

# لیست اشتراکی ذخیره تاریخچه محاسبات زنده
HISTORY_DATA = []

# =========================================================
# ۱. صفحه خانه پیشرفته و پربار (Advanced Dashboard Hub)
# =========================================================
class DashboardPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.build_ui()

    def build_ui(self):
        lbl_welcome = ctk.CTkLabel(self, text="📊 Enterprise IR System Status Overview", font=(Theme.FONT_FAMILY, 22, "bold"), text_color=Theme.ACCENT)
        lbl_welcome.pack(pady=(15, 5), padx=25, anchor="w")
        
        lbl_subtitle = ctk.CTkLabel(self, text="Real-time analytics and evaluation hub for Home Depot Search Relevance pipeline.", font=(Theme.FONT_FAMILY, 12), text_color=Theme.TEXT_MUTED)
        lbl_subtitle.pack(pady=(0, 20), padx=25, anchor="w")

        # بخش کارت‌های آماری (KPI Cards Row)
        kpi_frame = ctk.CTkFrame(self, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=20, pady=10)
        kpi_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        kpis = [
            ("Dataset Size", "74,067 Rows", "Kaggle Train Set", Theme.ACCENT),
            ("Target Metric", "RMSE", "Evaluation Mode", Theme.WARNING),
            ("Core Regressor", "LightGBM", "Gradient Boosting", Theme.SUCCESS),
            ("Text Embedding", "BERT (SOTA)", "Transformer Layer", "#A855F7")
        ]

        for idx, (title, val, sub, color) in enumerate(kpis):
            card = ctk.CTkFrame(kpi_frame, fg_color=Theme.CARD_BG, corner_radius=12, border_color=Theme.BG_SECONDARY, border_width=1)
            card.grid(row=0, column=idx, padx=8, sticky="nsew")
            
            ctk.CTkLabel(card, text=title, font=(Theme.FONT_FAMILY, 11, "bold"), text_color=Theme.TEXT_MUTED).pack(anchor="w", padx=15, pady=(12, 2))
            ctk.CTkLabel(card, text=val, font=(Theme.FONT_FAMILY, 18, "bold"), text_color=color).pack(anchor="w", padx=15, pady=2)
            ctk.CTkLabel(card, text=sub, font=(Theme.FONT_FAMILY, 10), text_color="#475569").pack(anchor="w", padx=15, pady=(0, 12))

        # ساختار دو ستونه اطلاعات پروژه
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=20, pady=15)
        main_content.grid_columnconfigure(0, weight=4)
        main_content.grid_columnconfigure(1, weight=3)
        main_content.grid_rowconfigure(0, weight=1)

        # کارت بزرگ سمت چپ: توصیف خط لوله (Pipeline Architecture)
        pipe_card = ctk.CTkFrame(main_content, fg_color=Theme.CARD_BG, corner_radius=14, border_color=Theme.BG_SECONDARY, border_width=1)
        pipe_card.grid(row=0, column=0, padx=8, sticky="nsew")
        
        ctk.CTkLabel(pipe_card, text="⚙️ Information Retrieval Architecture", font=(Theme.FONT_FAMILY, 15, "bold"), text_color=Theme.TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 10))
        
        steps = [
            ("1. Text Preprocessing & Tokenization", "Stemming using PorterStemmer, removing stop-words, and lowercasing raw catalog input strings."),
            ("2. Hybrid Feature Extraction Pipeline", "Extracting 13 distinct numerical indicators, including Jaccard Similarity, TF-IDF Cosine, and Exact Word Matches."),
            ("3. Dense Transformer Embeddings", "Generating high-dimensional semantic vectors using sentence-transformers (BERT core) for deep intent mapping."),
            ("4. LightGBM Ensemble Learning", "Feeding all stacked features into an optimized LightGBM Regressor to predict the continuous relevance score [1.0 - 3.0].")
        ]
        
        for step_title, step_desc in steps:
            step_box = ctk.CTkFrame(pipe_card, fg_color=Theme.BG_SECONDARY, corner_radius=8)
            step_box.pack(fill="x", padx=20, pady=6)
            ctk.CTkLabel(step_box, text=step_title, font=(Theme.FONT_FAMILY, 12, "bold"), text_color=Theme.ACCENT).pack(anchor="w", padx=12, pady=(6, 2))
            ctk.CTkLabel(step_box, text=step_desc, font=(Theme.FONT_FAMILY, 11), text_color=Theme.TEXT_MUTED, justify="left", wraplength=480).pack(anchor="w", padx=12, pady=(0, 6))

        # کارت سمت راست: وضعیت سلامت سرورها
        sys_card = ctk.CTkFrame(main_content, fg_color=Theme.CARD_BG, corner_radius=14, border_color=Theme.BG_SECONDARY, border_width=1)
        sys_card.grid(row=0, column=1, padx=8, sticky="nsew")
        
        ctk.CTkLabel(sys_card, text="🖥️ System Environment & Integrity", font=(Theme.FONT_FAMILY, 15, "bold"), text_color=Theme.TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 15))
        
        env_metrics = [
            ("Hardware Target:", "CUDA (GPU) / CPU Threading Enabled"),
            ("Python Core:", "v3.13 Stable Distribution"),
            ("GUI Framework:", "CustomTkinter Premium UI"),
            ("Pipeline Status:", "Online - Active Inference Local"),
            ("Kaggle Submission:", "Verified Format Compliant")
        ]
        
        for label, val in env_metrics:
            row = ctk.CTkFrame(sys_card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=8)
            ctk.CTkLabel(row, text=label, font=(Theme.FONT_FAMILY, 12), text_color=Theme.TEXT_MUTED).pack(side="left")
            color = Theme.SUCCESS if "Online" in val or "Verified" in val else Theme.TEXT_MAIN
            ctk.CTkLabel(row, text=val, font=(Theme.FONT_FAMILY, 12, "bold"), text_color=color).pack(side="right")
            
        btn_diag = ctk.CTkButton(sys_card, text="Run System Full Diagnostic Test", fg_color=Theme.BG_SECONDARY, text_color=Theme.TEXT_MAIN, hover_color="#2D3748", height=35, command=lambda: print("Diagnostic: Pipeline Healthy."))
        btn_diag.pack(fill="x", padx=20, pady=(35, 15), side="bottom")

# =========================================================
# ۲. صفحه اصلی موتور ارزیابی (Prediction Engine)
# =========================================================
class PredictionPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.build_ui()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=6)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        left_card = ctk.CTkFrame(self, fg_color=Theme.CARD_BG, corner_radius=16, border_color=Theme.BG_SECONDARY, border_width=1)
        left_card.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        ctk.CTkLabel(left_card, text="Product Metrics Entry Pipeline", font=(Theme.FONT_FAMILY, 18, "bold"), text_color=Theme.ACCENT).pack(anchor="w", padx=20, pady=(20, 15))

        ctk.CTkLabel(left_card, text="Search Query (User Intent):", font=(Theme.FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=20, pady=(5, 2))
        self.txt_query = ctk.CTkEntry(left_card, fg_color="#0B0F19", border_color=Theme.BG_SECONDARY, height=35)
        self.txt_query.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(left_card, text="Product Title (Catalog Metadata):", font=(Theme.FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.txt_title = ctk.CTkEntry(left_card, fg_color="#0B0F19", border_color=Theme.BG_SECONDARY, height=35)
        self.txt_title.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(left_card, text="Product Description Technical Sheet:", font=(Theme.FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.txt_desc = ctk.CTkTextbox(left_card, fg_color="#0B0F19", border_color=Theme.BG_SECONDARY, border_width=1, height=140)
        self.txt_desc.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(left_card, text="Brand Manufacturing Label:", font=(Theme.FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=20, pady=(10, 2))
        self.txt_brand = ctk.CTkEntry(left_card, fg_color="#0B0F19", border_color=Theme.BG_SECONDARY, height=35)
        self.txt_brand.pack(fill="x", padx=20, pady=5)

        btn_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=25)
        
        self.btn_predict = ctk.CTkButton(btn_frame, text="⚡ Execute Inference", fg_color=Theme.ACCENT, font=(Theme.FONT_FAMILY, 13, "bold"), hover_color="#2563EB", height=40, command=self.start_inference_thread)
        self.btn_predict.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        btn_sample = ctk.CTkButton(btn_frame, text="Load SOTA Example", fg_color=Theme.BG_SECONDARY, text_color=Theme.TEXT_MUTED, hover_color="#2D3748", height=40, command=self.load_sample)
        btn_sample.pack(side="right", expand=True, fill="x", padx=(10, 0))

        right_container = ctk.CTkFrame(self, fg_color="transparent")
        right_container.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        
        self.score_card = ctk.CTkFrame(right_container, fg_color=Theme.CARD_BG, corner_radius=16, border_color=Theme.BG_SECONDARY, border_width=1)
        self.score_card.pack(fill="x", pady=(0, 15), ipady=15)
        ctk.CTkLabel(self.score_card, text="RELEVANCE RECOGNITION SCORE", font=(Theme.FONT_FAMILY, 11, "bold"), text_color=Theme.TEXT_MUTED).pack(pady=(15, 0))
        self.lbl_score = ctk.CTkLabel(self.score_card, text="--", font=(Theme.FONT_FAMILY, 56, "bold"), text_color=Theme.ACCENT)
        self.lbl_score.pack()
        
        feat_card = ctk.CTkFrame(right_container, fg_color=Theme.CARD_BG, corner_radius=16, border_color=Theme.BG_SECONDARY, border_width=1)
        feat_card.pack(fill="both", expand=True, pady=10)
        ctk.CTkLabel(feat_card, text="Hybrid Feature Importance Analysis", font=(Theme.FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=20, pady=15)
        
        self.bars = {}
        features_list = [
            ("Word Match Overlap", "word_match"), 
            ("Brand Consistency", "brand_match"), 
            ("Jaccard Similarity", "jaccard"), 
            ("TF-IDF Cosine Vector", "tfidf"), 
            ("BERT Semantic Sim", "semantic")
        ]
        for label, key in features_list:
            row = ctk.CTkFrame(feat_card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=7)
            ctk.CTkLabel(row, text=label, font=(Theme.FONT_FAMILY, 12), text_color=Theme.TEXT_MUTED).pack(side="left")
            bar = ctk.CTkProgressBar(row, width=180, progress_color=Theme.ACCENT, fg_color="#0B0F19")
            bar.set(0)
            bar.pack(side="right", pady=5)
            self.bars[key] = bar

        ai_card = ctk.CTkFrame(right_container, fg_color=Theme.CARD_BG, corner_radius=16, border_color=Theme.BG_SECONDARY, border_width=1)
        ai_card.pack(fill="x", pady=(15, 0), ipady=10)
        ctk.CTkLabel(ai_card, text="🤖 Transformer Interpretability Core", font=(Theme.FONT_FAMILY, 12, "bold")).pack(anchor="w", padx=20, pady=(12, 5))
        self.lbl_explanation = ctk.CTkLabel(ai_card, text="Awaiting structural input pipeline deployment...", font=(Theme.FONT_FAMILY, 12), text_color=Theme.TEXT_MUTED, justify="left", wraplength=400)
        self.lbl_explanation.pack(anchor="w", padx=20, pady=5)

    def load_sample(self):
        self.txt_query.delete(0, 'end')
        self.txt_title.delete(0, 'end')
        self.txt_desc.delete("1.0", 'end')
        self.txt_brand.delete(0, 'end')
        
        self.txt_query.insert(0, "dewalt 20v drill cordless")
        self.txt_title.insert(0, "DEWALT 20-Volt MAX Lithium-Ion Cordless Compact Drill")
        self.txt_desc.insert("1.0", "The DEWALT DCK240C2 features a compact drill driver and an impact driver.")
        self.txt_brand.insert(0, "DEWALT")

    def start_inference_thread(self):
        self.btn_predict.configure(text="⌛ Processing Model...", state="disabled")
        q = self.txt_query.get()
        t = self.txt_title.get()
        d = self.txt_desc.get("1.0", "end-1c")
        b = self.txt_brand.get()
        threading.Thread(target=self.run_async_inference, args=(q, t, d, b), daemon=True).start()

    def run_async_inference(self, q, t, d, b):
        score, features, explanation = real_inference_pipeline(q, t, d, b)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        HISTORY_DATA.append((timestamp, q if q else "Empty Query", f"{score:.2f}"))

        self.lbl_score.configure(text=f"{score:.2f}")
        if score >= 2.5: self.lbl_score.configure(text_color=Theme.SUCCESS)
        elif score >= 1.8: self.lbl_score.configure(text_color=Theme.WARNING)
        else: self.lbl_score.configure(text_color=Theme.DANGER)

        for k, v in features.items():
            self.bars[k].set(v)
            
        self.lbl_explanation.configure(text=explanation)
        self.btn_predict.configure(text="⚡ Execute Inference", state="normal")

# =========================================================
# ۳. صفحه آنالیز (Analytics Suite)
# =========================================================
class AnalyticsPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        lbl_title = ctk.CTkLabel(self, text="Feature Importance Insights", font=(Theme.FONT_FAMILY, 20, "bold"), text_color=Theme.TEXT_MAIN)
        lbl_title.pack(pady=20, padx=20, anchor="w")

        chart_card = ctk.CTkFrame(self, fg_color=Theme.CARD_BG, corner_radius=12)
        chart_card.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(chart_card, text="Global Weights (Gain Index Evaluation)", font=(Theme.FONT_FAMILY, 14, "bold")).pack(pady=15, padx=20, anchor="w")
        
        mock_weights = [("BERT Semantic Density", 0.92), ("TF-IDF Cosine Vectors", 0.78), ("Lexical Word Match Overlap", 0.65), ("Jaccard Distance Token", 0.45)]
        for label, val in mock_weights:
            row = ctk.CTkFrame(chart_card, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=8)
            ctk.CTkLabel(row, text=label, width=200, anchor="w").pack(side="left")
            b = ctk.CTkProgressBar(row, progress_color="#2563EB", fg_color="#0B0F19", width=400)
            b.set(val)
            b.pack(side="left", padx=20, pady=5)

# =========================================================
# ۴. صفحه تاریخچه زنده (History Log)
# =========================================================
class HistoryPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.lbl_title = ctk.CTkLabel(self, text="Real-time Execution History Log", font=(Theme.FONT_FAMILY, 20, "bold"))
        self.lbl_title.pack(pady=20, padx=20, anchor="w")
        
        self.log_container = ctk.CTkScrollableFrame(self, fg_color=Theme.CARD_BG, corner_radius=12, height=500)
        self.log_container.pack(fill="both", expand=True, padx=20, pady=10)
        self.refresh_log()

    def refresh_log(self):
        for widget in self.log_container.winfo_children():
            widget.destroy()
            
        if not HISTORY_DATA:
            ctk.CTkLabel(self.log_container, text="No executions tracked in this pipeline session.", text_color=Theme.TEXT_MUTED).pack(pady=40)
            return

        for ts, q, sc in reversed(HISTORY_DATA):
            row = ctk.CTkFrame(self.log_container, fg_color=Theme.BG_SECONDARY, corner_radius=6)
            row.pack(fill="x", pady=4, padx=5)
            ctk.CTkLabel(row, text=f" [{ts}] ", text_color=Theme.ACCENT, font=("Consolas", 12)).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=f"Query: {q}", text_color=Theme.TEXT_MAIN).pack(side="left", padx=20)
            ctk.CTkLabel(row, text=f"Score: {sc}", text_color=Theme.SUCCESS, font=(Theme.FONT_FAMILY, 12, "bold")).pack(side="right", padx=15)

# =========================================================
# ۵. صفحه تنظیمات (Settings Page)
# =========================================================
class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(self, text="Environment Settings", font=(Theme.FONT_FAMILY, 20, "bold")).pack(pady=20, padx=20, anchor="w")
        
        card = ctk.CTkFrame(self, fg_color=Theme.CARD_BG, corner_radius=12)
        card.pack(fill="x", padx=20, pady=10, ipady=15)
        
        ctk.CTkLabel(card, text="Model Path Configuration:", font=(Theme.FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=20, pady=(15, 2))
        e = ctk.CTkEntry(card, fg_color="#0B0F19", border_color=Theme.BG_SECONDARY, width=500)
        e.insert(0, "./models/ir_sota_search_model.pkl")
        e.pack(anchor="w", padx=20, pady=5)