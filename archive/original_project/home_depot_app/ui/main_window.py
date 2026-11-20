# ui/main_window.py
import customtkinter as ctk
from theme import Theme
from ui.sidebar import Sidebar
from ui.pages import DashboardPage, PredictionPage, AnalyticsPage, HistoryPage, SettingsPage

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Home Depot Search Relevance Engine - Premium SOTA Suite")
        self.geometry("1400x900")
        self.configure(fg_color=Theme.BG_PRIMARY)
        self.build_ui()

    def build_ui(self):
        # تاپ‌بار بالا
        top_bar = ctk.CTkFrame(self, fg_color=Theme.BG_PRIMARY, height=70, corner_radius=0, border_color=Theme.BG_SECONDARY, border_width=1)
        top_bar.pack(fill="x", side="top")
        
        lbl_logo = ctk.CTkLabel(top_bar, text="🧡 Home Depot Search Relevance Enterprise Dashboard", font=(Theme.FONT_FAMILY, 15, "bold"))
        lbl_logo.pack(side="left", padx=25, pady=20)
        
        lbl_status = ctk.CTkLabel(top_bar, text="● LightGBM: Ready | ● BERT: Active | Threads: Stable", font=(Theme.FONT_FAMILY, 12, "bold"), text_color=Theme.SUCCESS)
        lbl_status.pack(side="right", padx=25, pady=20)

        # بدنه
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True)

        # سایدبار
        self.sidebar = Sidebar(self.body, self.set_page)
        self.sidebar.pack(fill="y", side="left")

        # کانتینر صفحات
        self.container = ctk.CTkFrame(self.body, fg_color="transparent")
        self.container.pack(fill="both", expand=True, side="right", padx=10, pady=10)

        # ثبت دقیق و کامل تمام صفحات داینامیک جدید
        self.pages = [
            DashboardPage(self.container),
            PredictionPage(self.container),
            AnalyticsPage(self.container),
            HistoryPage(self.container),
            SettingsPage(self.container)
        ]
        
        self.set_page(1) # شروع مستقیم روی بخش پردیکشن

    def set_page(self, index):
        for page in self.pages:
            page.pack_forget()
        
        # فرآیند رندر و نمایش صفحه کلیک شده
        self.pages[index].pack(fill="both", expand=True)
        if hasattr(self.pages[index], "refresh_log"):
            self.pages[index].refresh_log()