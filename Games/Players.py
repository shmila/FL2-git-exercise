import Result
import random
from game import FizzGame
import sys


class Player(object):
    def __init__(self, name = ''):
        self.name = name
        self.game_history = Result.GameHistory(name)

    def play(self,curr_num):
        raise NotImplementedError




class HumanPlayer(Player):
    def __init__(self,name=''):
        super(HumanPlayer,self).__init__(name)

    def play(self,curr_num):
        move = raw_input('%s, Please insert your next move:\n' % self.name)
        return move

class ComputerPlayer(Player):
    def __init__(self, name='comp1', level=1):
        super(ComputerPlayer, self).__init__(name)
        #level should be between 1 (easy) to 3 (hard)
        if level < 1:
            self.level = 1
        elif level > 3:
            self.level = 3
        else:
            self.level = level

    def play(self,curr_num):
        correct = FizzGame.get_correct_answer(curr_num)
        # decide if computer is going to be right in this turn according to its level
        is_correct = self.level >= random.randint(1,3)
        ans = None
        if not is_correct:
            curr_options = [str(curr_num)] + FizzGame.options
            rand_opt = curr_options[random.randint(0,len(curr_options)-1)]
            while (rand_opt == correct):
                rand_opt = curr_options[random.randint(0, len(curr_options)-1)]
            ans =  rand_opt
        else:
            ans = correct
        print('%s plays: %s' % (self.name,ans))
        return ans

class TicTacPlayer(Player):
    def __init__(self,name=''):
        super(TicTacPlayer,self).__init__(name)

    def play(self,player, board):
        board_size = len(board)
        move = raw_input(
            "%s - please enter your move in the format [ROW] [COL]. where the indices are in the range 0-%d: " % (
            self.name, board_size - 1))
        while not self.isLegal(move, board):
            print "Illegal move"
            move = raw_input(
                "%s - please enter your move in the format [ROW][COL]. where the indices are in the range 0-%d: " % (
                self.name, board_size - 1))
        splitted = move.split(' ')
        row = int(splitted[0])
        col = int(splitted[1])
        board[row][col] = player
        self.printBoard(board)

    def isLegal(self, move, board):
        board_size = len(board)
        splitted = move.split(' ')
        if len(splitted) is not 2:
            return False
        else:
            row = int(splitted[0])
            col = int(splitted[1])
            if row < 0 or row >= board_size or col < 0 or col >= board_size:
                return False
            else:
                if not (board[row][col] == 0):
                    return False
                else:
                    return True

    def printBoard(self,board):
        board_size = len(board)
        print "The Board:"
        for i in range(board_size * 2 + 1):
            for j in range(board_size):
                if i % 2 == 0:
                    sys.stdout.write("--")
                    continue
                if (j == 0):
                    sys.stdout.write("|")
                sys.stdout.write(TicTacPlayer.getSign(board[i / 2][j]) + "|")
            print('')

    @staticmethod
    def getSign(param):
        if param == 1:
            return 'X'
        elif param == 2:
            return 'O'
        else:
            return ' '
