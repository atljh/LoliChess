<h1>Chess Board Image Recognition and Move Prediction</h1>

This project uses computer vision techniques and a trained neural network to recognize the chessboard in a given image, predict the positions of the chess pieces, and then use the Stockfish chess engine to suggest the best move for the next player.
Requirements

    Python 3.x
    OpenCV (cv2)
    Numpy
    Tensorflow
    Stockfish
    subprocess

Installation

    Clone the repository:

    bash

$ git clone https://github.com/atljh/LoliChess.git
$ cd LoliChess

Install the required dependencies:

bash

    $ pip install opencv-python numpy tensorflow python-chess

    Download the Stockfish chess engine from stockfishchess.org/download/ and copy the binary file to the project directory.

Usage

    To run the program, execute the following command:

    bash

$ python main.py

The program will wait for you to provide an image file path. Once the path is provided, it will recognize the chessboard, predict the positions of the pieces, and suggest the best move for the next player.