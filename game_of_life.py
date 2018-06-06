import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

class GameOfLife:

    def __init__(self,rows=3,cols=3,prob=0.5):
        self.board = self.initBoard(rows,cols,prob)
        self.p = None

    def initBoard(self,rows,cols,prob):
        board = np.zeros((rows,cols,3))
        mask = np.random.rand(rows,cols,3) > prob
        board[mask] = 1
        return board

    def play_animation(self):

        fig = plt.figure()
        self.p = plt.imshow(self.board)
        ani = animation.FuncAnimation(fig, self.update)
        plt.show()

    def play(self):
        #need to change it for supporting 3 channels
        rows, cols = self.board.shape
        plt.matshow(self.board)

        while True:
            # eval cells
            for i in range(rows):
                for j in range(cols):
                    self.eval_cell(i,j)

            # plot and show fig
            plt.matshow(self.board,fignum=False)
            # sleep
            plt.pause(1)

    def update(self,frame_num):
        rows, cols, clr = self.board.shape
        # eval cells
        for i in range(rows):
            for j in range(cols):
                for k in range(clr):
                    self.eval_cell(i, j, k)
        self.p.set_data(self.board)
        return [self.p]

    def eval_cell(self, row, col, color):
        rows , cols, colors = self.board.shape
        num_alive = 0
        start_row = row - 1
        end_row = row + 1
        start_col = col - 1
        end_col = col + 1
        if row == 0:
            start_row += 1
        if row == rows -1:
            end_row -= 1
        if col == 0:
            start_col += 1
        if col == cols -1:
            end_col -= 1
        for i in range(start_row,end_row+1):
            for j in range(start_col,end_col+1):
                for k in range(colors):
                    if self.board[i,j,k] == 1:
                        # found alive cell
                        # check if this is the target cell
                        if (i==row and j==col and k==color) or k!=color:
                            continue
                        else:
                            num_alive += 1
        if self.board[row,col,color] == 1:
            # cell alive
            if num_alive < 2 or num_alive > 3:
                # make him dead
                self.board[row,col,color] = 0
        else:
            # cell is dead
            if num_alive == 3:
                self.board[row,col,color] = 1


game = GameOfLife(55,55,0.5)
# game.play()
game.play_animation()