# main.py
import os
import sys

# چک کردن خودکار پکیج فرانت روی سیستم کاربر
try:
    import customtkinter as ctk
except ImportError:
    print("Installing customtkinter framework...")
    os.system(f'"{sys.executable}" -m pip install customtkinter')
    import customtkinter as ctk

from ui.main_window import MainWindow

if __name__ == "__main__":
    # تنظیم پوسته سراسری دسکتاپ به تم تاریک پرمیوم
    ctk.set_appearance_mode("Dark")
    
    app = MainWindow()
    app.mainloop()