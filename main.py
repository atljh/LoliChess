import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import pyautogui
from pathlib import Path
from datetime import datetime
import time
from stockfish import Stockfish


ROOT_DIR = Path(__file__).resolve(strict=True).parent
SCREENSHOTS_DIR = str(ROOT_DIR / "screenshots")


stockfish = Stockfish(path="")

def show(cap, img, time=1000):
    cv2.imshow(cap, img)
    cv2.waitKey(time)

# обрезка под размеры доски
def cut_to_size_board(img, cnts):
    #print("Всего контуров на исходном: ", len(cnts))
    for cnt in cnts:
        answer = is_board(cnt)
        try:
            is_b, sqr, x, y, w, h = answer
        except ValueError:
            is_b, *_ = answer
            
        if is_b:
            #print(f"sqr = {sqr}, coords: {x}, {y}, w:{w},h: {h}")
            cropped_img = img[y+1:y+h-1, x+1:x+w-1]
            return cropped_img
    return img

# отрисовка контуров
def show_contours(img, cnts):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
    return img

# явлиется ли контур доской
def is_board(cnt):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True) #апроксимация(упрощение формы)
    sqr = cv2.contourArea(cnt)
    if len(approx) == 4 and sqr>=img_sqr/7: #если углов 4 и площадь >= чем 1/7 от исходного изображения
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.98 and aspect_ratio <= 1.02: # если соотношение = квадрат+-2% - то скорее всего это доска
            return [True, sqr, x, y, w, h]
    return [False]


screenshot_name = str(datetime.now().strftime('%d%m%Y%H%M%S'))

if sys.platform.startswith("linux"):
    # os.system(f"gnome-screenshot -w -f screenshots/{screenshot_name}.png")
    ...
elif sys.platform == "darwin":
    ...
elif os.name == "nt":
    image = pyautogui.screenshot()

# img_path = f'{SCREENSHOTS_DIR}/{screenshot_name}.png'
img_path = f'{SCREENSHOTS_DIR}/white.png'


img = cv2.imread(img_path)

h, w, _ = img.shape
img_sqr = h*w


# в серый одноканальный 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Поиск контуров на бинаризованном изображении
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Поиск контура доски и обрезка изображение под ее размер

# board - изображение доски с фигурами, show("name", board, time) для просмотра 
board = cut_to_size_board(img, contours)

#show('board', board, 2000)



# модель нейронки

model = tf.keras.models.load_model('/home/donbuasson/Music/LoliChess/chess_detector.h5')

# размеры доски
bh,bw = board.shape[:2]

# размеры ячейки
cw, ch = bw//8, bh//8

# список для картинок ячеек
images64 = []

figures1=[ '1', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']

# нарезка board на 64 ячейки
for cellY in range(8):
    for cellX in range(8):
        cropped_img = board[ch*cellY:ch*(cellY+1), ch*cellX:ch*(cellX+1)]
        # in gray
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # in 50*50
        resized_gray = cv2.resize(gray, (50,50))
        # reshape array in 1 layer
        img = resized_gray.reshape((50, 50, 1))
        img = img.astype('float32')

        # добавление картинки в список
        images64.append(img)
        

images64 = np.array(images64)
fen = ""

answer = model.predict(images64)

# структурировать ответы нейронки в FEN код
tmp = 0
for i, a in enumerate(answer):      # 11R11111 111111PP PK111P11 11P1Pp11 p1r1p111 1N1pb111 111111pp 1k111111
    symbol = figures1[np.argmax(a)] # 2R5/6PP/PK3P2/2P1Pp2/p1r1p3/1N1pb3/6pp/1k6
    if (i+1)%8 == 0:                
        if symbol == "1":
            tmp += 1
            fen += str(tmp)+'/'
            tmp = 0
        elif tmp > 0:
            fen += str(tmp)+symbol+'/'
            tmp = 0
        else:
            fen += symbol + '/'
    else:
        if symbol == "1":
            tmp += 1
        elif tmp > 0:
            fen += str(tmp) + symbol
            tmp = 0
        else:
            fen += symbol


print(f"\n\nFEN: {fen[0:-1]}\n\n")
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2"
# fen = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
# stockfish.set_fen_position(f"{fen} w KQkq - 0 1")
stockfish.set_fen_position(fen)
print(stockfish.is_fen_valid(fen))
print(stockfish.get_best_move())
print(stockfish.get_board_visual())
# show("res", board, 2000)
time.sleep(10)
# os.remove(f'{SCREENSHOTS_DIR}/{screenshot_name}.png')