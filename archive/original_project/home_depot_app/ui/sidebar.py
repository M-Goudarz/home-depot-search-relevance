# ui/sidebar.py
import customtkinter as ctk
from theme import Theme

class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, change_page_callback):
        super().__init__(parent, fg_color=Theme.CARD_BG, corner_radius=0, width=220)
        self.change_page = change_page_callback
        self.build_ui()

    def build_ui(self):
        lbl_nav = ctk.CTkLabel(self, text="CORE NAVIGATOR", font=(Theme.FONT_FAMILY, 11, "bold"), text_color="#475569")
        lbl_nav.pack(pady=(30, 20), padx=20, anchor="w")

        nav_items = [
            ("Dashboard", 0),
            ("Prediction Engine", 1),
            ("Analytics Suite", 2),
            ("History Log", 3),
            ("Settings", 4)
        ]

        self.buttons = []
        for text, index in nav_items:
            btn = ctk.CTkButton(
                self, text=f"  {text}", 
                font=(Theme.FONT_FAMILY, 13),
                fg_color="transparent", text_color=Theme.TEXT_MUTED,
                hover_color=Theme.BG_SECONDARY, anchor="w", height=40,
                command=lambda idx=index: self.handle_click(idx)
            )
            btn.pack(fill="x", padx=10, pady=4)
            self.buttons.append(btn)
        
        # صفحه Prediction Engine به صورت پیش‌فرض هایلایت شود
        self.buttons[1].configure(fg_color=Theme.BG_SECONDARY, text_color=Theme.ACCENT, font=(Theme.FONT_FAMILY, 13, "bold"))

    def handle_click(self, index):
        for btn in self.buttons:
            btn.configure(fg_color="transparent", text_color=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 13))
        
        self.buttons[index].configure(fg_color=Theme.BG_SECONDARY, text_color=Theme.ACCENT, font=(Theme.FONT_FAMILY, 13, "bold"))
        self.change_page(index)