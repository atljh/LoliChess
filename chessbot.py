import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from stockfish import Stockfish
import subprocess
# import pyautogui
import time


STOCKFISH_PATH = '/home/donbuasson/Music/LoliChess/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64'
MODEL_PATH = '/home/donbuasson/Music/LoliChess/chess_detector.h5'

# Load model and stockfish engine
model = tf.keras.models.load_model(MODEL_PATH)
stockfish = Stockfish(STOCKFISH_PATH)


def cut_to_size_board(img, cnts, img_sqr):
    """Crop the image to the size of the chessboard."""
    for cnt in cnts:
        answer = is_board(cnt, img_sqr)
        try:
            is_b, sqr, x, y, w, h = answer
        except ValueError:
            is_b, *_ = answer
            
        if is_b:
            cropped_img = img[y+1:y+h-1, x+1:x+w-1]
            return cropped_img
    raise ValueError('Board not found')


def show_contours(img, cnts):
    """Draw contours on the image."""
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
    return img


def is_board(cnt, img_sqr):
    """Check if the contour is a chessboard."""
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    sqr = cv2.contourArea(cnt)
    if len(approx) == 4 and sqr>=img_sqr/7:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.98 and aspect_ratio <= 1.02:
            return [True, sqr, x, y, w, h]
    return [False]


def board_to_cells(board):
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



def generate_fen(model_answer, figures_names, next_move) -> str:
    fen = ""
    tmp = 0
    for i, a in enumerate(model_answer):
        symbol = figures_names[np.argmax(a)]
        if (i+1) % 8 == 0:
            if symbol == "1":
                tmp += 1
            elif tmp > 0:
                fen += str(tmp)+symbol
                tmp = 0
            else:
                fen += symbol
            if i != 63:
                fen += '/'
        else:
            if symbol == "1":
                tmp += 1
            elif tmp > 0:
                fen += str(tmp)+symbol
                tmp = 0
            else:
                fen += symbol
    colour = 'w' if next_move else 'b'
    fen += f' {colour} KQkq - 0 1'
    return fen


def get_best_move(img, last_fen, next_move):
    """
    Detects the chessboard in the input image and returns the best move using Stockfish engine.
    
    Args:
        img (numpy.ndarray): Input image of the chessboard.
        last_fen (str): The last FEN notation of the chessboard.
        next_move (bool): True if it is white's turn, False if it is black's turn.
        
    Returns:
        tuple: A tuple containing the new FEN notation and the boolean indicating the next move.
    """
    
    # Calculate the image dimensions
    height, width, _ = img.shape
    img_sqr = height * width

    # Define the names of the chess figures
    figures_names = ['1', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
    
    # Convert image to grayscale and apply thresholding to detect the chessboard
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 120, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Cut the image to the size of the chessboard
    try:
        board = cut_to_size_board(img, contours, img_sqr)
    except Exception as e:
        raise ValueError("Chessboard not found in the image.") from e
    
    # Convert the board to individual cell images and predict the type of each cell
    images64 = np.array(board_to_cells(board))
    predictions = model.predict(images64, verbose=0)
    
    # Generate the new FEN notation based on the predictions and the current player
    fen_notation = generate_fen(predictions, figures_names, next_move)

    # Check if the new FEN notation is valid and get the best move from Stockfish engine
    if not stockfish.is_fen_valid(fen_notation):
        raise ValueError("Invalid FEN notation.")
    
    stockfish.set_fen_position(fen_notation)
    try:
        best_move = stockfish.get_best_move()
        visual = stockfish.get_board_visual()
    except Exception as e:
        raise ValueError("Error occurred while getting the best move from Stockfish engine.") from e
    
    # Determine the player of the next move and return the new FEN notation and player
    if last_fen[:-13] == fen_notation[:-13]:
        return fen_notation, next_move
    next_move = not next_move
    return fen_notation, next_move



# Загрузка изображения
def get_best_move(img, last_fen, next_move):

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
        print('Board not found')
        sys.exit(0)
            
    images64 = np.array(board_to_cells(board))
    preditctions = model.predict(images64, verbose=0)
    FEN = generate_fen(preditctions, figures_names, next_move)

    # print(stockfish.is_fen_valid(FEN))
    stockfish.set_fen_position(FEN)
    try:
        best_move = stockfish.get_best_move()
        visual = stockfish.get_board_visual()
    except Exception as e:
        print(e)
    if last_fen[:-13] == FEN[:-13]:
        print(best_move, next_move)    
        return FEN, next_move
    next_move = not next_move
    print(best_move, next_move)    
    print(visual)
    # print(FEN)
    last_fen = FEN
    return last_fen, next_move


def main():
    # colour = input()
    # if colour not in ['w', 'b']:
    #     print('w - for white, b - for black')
    #     return
    name = '.frame.png'
    LAST_FEN = 'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'
    NEXT_MOVE = True
    while True:
        subprocess.run(['gnome-screenshot','--display=:0', '-f', f'{name}'])
        # image = pyautogui.screenshot()
        # image.save(name)
        frame = cv2.imread(name)
        frame = np.array(frame)
        last_fen, next_move = get_best_move(frame, LAST_FEN, NEXT_MOVE)
        time.sleep(1)
        NEXT_MOVE = next_move
        LAST_FEN = last_fen


if __name__ == '__main__':
    main()