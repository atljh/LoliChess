import tkinter as tk
import cv2
import subprocess
import screen2fen
from PIL import Image, ImageTk, ImageDraw, ImageFont


frame_path = 'LoliChess/temp/.frame.png'

def update(side):
    sc_take()
    frame = cv2.imread(frame_path)
    answer = screen2fen.main(frame, side)

    canvas = image.copy()
    draw_text(canvas, answer)
    tk_image = ImageTk.PhotoImage(canvas)
    
    label.configure(image=tk_image)
    label.image = tk_image

def sc_take():    
    subprocess.run(['gnome-screenshot', '-f', f'{frame_path}'])

def draw_text(canvas, text='Example text'):
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("Chess/fonts/Montserrat-Bold.ttf", 36)  # указываем шрифт и размер
    draw.text((10, 40), text, font=font, fill=(0, 0, 0)) 

# Создание окна
root = tk.Tk()
root.geometry("500x300")

# Создание полотна
image = Image.new('RGB', (300, 200), (255, 255, 255))

tk_white = ImageTk.PhotoImage(image)
# Размещение виджета Label
label = tk.Label(root, image=tk_white)
label.pack()

# Создание кнопки
button = tk.Button(root, text="Move for white")
button1 = tk.Button(root, text="Move for black")

# Размещение кнопки на окне
button.pack()
button1.pack()

# Функция обработки нажатия кнопки
def white():
    update(' w')

def black():
    update(' b')

# Привязка обработчика к кнопке
button.config(command=white)
button1.config(command=black)

root.mainloop()