import os
import cv2
import numpy as np
import tensorflow as tf
from stockfish import Stockfish
import subprocess
import pyautogui
import time


stockfish = Stockfish('/home/user/Music/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64')
model = tf.keras.models.load_model('/home/user/chess_detector.h5')

# просмотр картинки 
def show(cap, img, time=1000):
    cv2.namedWindow(cap, cv2.WINDOW_NORMAL)
    cv2.imshow(cap, img)
    cv2.waitKey(time)

# обрезка под размеры доски
def cut_to_size_board(img, cnts, img_sqr):
    #print("Всего контуров на исходном: ", len(cnts))
    for cnt in cnts:
        answer = is_board(cnt, img_sqr)
        try:
            is_b, sqr, x, y, w, h = answer
        except ValueError:
            is_b, *_ = answer
            
        if is_b:
            #tmp_img=cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
            #show('contrs', tmp_img, 0)
            print(f"sqr = {sqr}, coords: {x}, {y}, w:{w},h: {h}")
            cropped_img = img[y+1:y+h-1, x+1:x+w-1]
            return cropped_img
    raise Exception("BoardNotFound")

# отрисовка контуров
def show_contours(img, cnts):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
    return img

# явлиется ли контур доской
def is_board(cnt, img_sqr):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True) #апроксимация(упрощение формы)
    sqr = cv2.contourArea(cnt)
    if len(approx) == 4 and sqr>=img_sqr/7: #если углов 4 и площадь >= чем 1/7 от исходного изображения
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.98 and aspect_ratio <= 1.02: # если соотношение = квадрат+-2% - то скорее всего это доска
            return [True, sqr, x, y, w, h]
    return [False]

# нарезка board на 64 ячейки
def board2cells(board):
    # размеры доски
    bh,bw = board.shape[:2]
    # размеры ячейки
    cw, ch = bw//8, bh//8
    images64 = []
    for cellY in range(8):
        for cellX in range(8):
            cropped_img = board[cw*cellY:cw*(cellY+1), ch*cellX:ch*(cellX+1)]
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (50,50))
            img = resized_gray.reshape((50, 50, 1))
            img = img.astype('float32')

            # добавление картинки в список
            images64.append(img)
    return images64
            

# структурировать ответы нейронки в FEN код
def pred2FEN(model_answer, figures_names, colour) -> str:
    fen = ""
    tmp = 0
    for i, a in enumerate(model_answer):      
        symbol = figures_names[np.argmax(a)]  
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
    fen = fen[0:-1]
    fen += f' {colour} KQkq - 0 1'
    return fen


# Загрузка изображения
def main(img, colour, last_fen):

    #img = cv2.imread(f'{screnshots_folder}/{screenshots[-1]}')
    h, w, _ = img.shape
    img_sqr = h*w

    figures_names=[ '1', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
    
    # в серый одноканальный, бинаризация изображения, поиск контуров на бинаризованном изображении
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 120, 120, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Поиск контура доски и обрезка изображение под ее размер
    try:
        board = cut_to_size_board(img, contours, img_sqr)
    except Exception as e:
        return "Board not found..."
    
    # модель нейронки

    images64 = np.array(board2cells(board))
    preditctions = model.predict(images64)
    FEN = pred2FEN(preditctions, figures_names, colour)

    # print(stockfish.is_fen_valid(FEN))
    stockfish.set_fen_position(FEN)
    try:
        best_move = stockfish.get_best_move()
        visual = stockfish.get_board_visual()
    except Exception as e:
        ...
    if last_fen == FEN:
        print(best_move)    
        return FEN
    print(best_move)    
    print(visual)
    # print(FEN)
    last_fen = FEN
    return last_fen




def start():
    colour = input()
    if colour not in ['w', 'b']:
        print('w - for white, b - for black')
        return
    name = '.frame.png'
    LAST_FEN = None
    while True:
        # subprocess.run(['gnome-screenshot','--display=:0', '-f', f'{name}'])
        image = pyautogui.screenshot()
        image.save('.frame.png')
        frame = cv2.imread(name)
        frame = np.array(frame)
        last_fen = main(frame, colour, LAST_FEN)
        time.sleep(3)
        LAST_FEN = last_fen


start()