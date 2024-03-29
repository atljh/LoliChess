import os
import sys
from os.path import join, dirname
from typing import List, Tuple, Optional, Any


import cv2
import pyautogui
import subprocess
import numpy as np
import tensorflow as tf
from   stockfish import Stockfish
from   dotenv import load_dotenv


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH')
MODEL_PATH     = os.environ.get('MODEL_PATH')

model     = tf.keras.models.load_model(MODEL_PATH)
stockfish = Stockfish(STOCKFISH_PATH, depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30})


def cut_to_size_board(img: np.ndarray, cnts: List[Any], img_sqr: int) -> Optional[np.ndarray]:
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
    return []


def is_board(cnt: Any, img_sqr: int) -> List[Optional[Any]]:
    """Check if the contour is a chessboard"""
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    sqr = cv2.contourArea(cnt)
    if len(approx) == 4 and sqr>=img_sqr/7:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.98 and aspect_ratio <= 1.02:
            return [True, sqr, x, y, w, h]
    return [False]


def board_to_cells(board: np.ndarray) -> List[np.ndarray]:
    bh,bw = board.shape[:2]
    cw, ch = bw//8, bh//8
    images64 = []
    for cellY in range(8):
        for cellX in range(8):
            cropped_img = board[cw*cellY:cw*(cellY+1), ch*cellX:ch*(cellX+1)]
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (50,50))
            img = resized_gray.reshape((50, 50, 1))
            img = img.astype('float32')
            images64.append(img)
    return images64


def generate_fen(predictions: List[np.ndarray], next_move: bool, color: str, P) -> Tuple[str, str]:
    figures_names = ['1', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
    fen = ""
    tmp = 0
    for i, a in enumerate(predictions):      
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
    if color == 'b':
        fen = fen[::-1]
    color = 'w' if next_move else 'b'
    fen += f' {color} KQkq - 0 1'
    return fen, color


def get_best_move(img: np.ndarray, color: str, last_fen: str, next_move: bool) -> Tuple[str, bool]:

    height, width, _ = img.shape
    img_sqr = height * width
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 120, 120, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    board = cut_to_size_board(img, contours, img_sqr)
    if not len(board):
        print('Board not found')
        return main()
    
    images64 = np.array(board_to_cells(board))
    predictions = model.predict(images64, verbose=0)
    
    fen_notation, move_color = generate_fen(predictions, next_move, color, last_fen)

    if not stockfish.is_fen_valid(fen_notation):
        print(fen_notation)
        raise ValueError("Invalid FEN notation. Did you choose your color right?")
    
    stockfish.set_fen_position(fen_notation)
    try:
        best_move = stockfish.get_best_move_time(1000)
        if color == 'w':
            visual = stockfish.get_board_visual()
        else:
            visual = stockfish.get_board_visual(False)
    except Exception as e:
        raise ValueError("Error occurred while getting the best move from Stockfish engine.") from e
    if (last_fen == fen_notation):
        return fen_notation, next_move

    actions = stockfish.get_evaluation()
    if actions.get('type') == 'mate':
        print(f'Mate in {abs(actions["value"])}')

    if (last_fen[:-13] != fen_notation[:-13]) and len(last_fen) > 0:
        next_move = not next_move
        return fen_notation, next_move
    
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

    print('\n', visual)
    print(f'Best move for {move_color}:', best_move)
    return fen_notation, next_move


def get_screenshot(name: str ='.frame.png') -> str:
    if os.getenv('XDG_SESSION_TYPE') == 'wayland':
        subprocess.run(['gnome-screenshot', '--display=:0', '-f', f'{name}'])
    elif os.getenv('XDG_SESSION_TYPE') == 'x11':
        image = pyautogui.screenshot()
        image.save(name)
    return name


def main() -> None:
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

    color = input('Enter your color (w, b): ')
    if color.lower() not in ['w', 'b']:
        print('w - for white, b - for black')
        return
    
    if color == 'w':
        _next_move = True
    else:
        _next_move = False
    
    # _last_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'
    _last_fen = ''
    while True:
        name = get_screenshot()
        frame = cv2.imread(name)
        img = np.array(frame)

        last_fen, next_move = get_best_move(img, color, _last_fen, _next_move)
        _next_move = next_move
        _last_fen = last_fen

        #TODO: circular

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' \nStopped')
        sys.exit(130)