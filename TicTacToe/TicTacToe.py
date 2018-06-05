import sys
sys.path.insert(0, '/home/drax/PycharmProjects/Practice Python/warCard')
from Player import Player
from scipy.signal.ltisys import bode
# foo = imp.load_source('Player.name', '/home/drax/PycharmProjects/Practice Python/warCard/Player.py')
# foo.MyClass()

# from Player import Player


class TicTacToe:
    BOARD_SIZE = 3
    ROWS = 1
    COLS = 0

    def get_player_request(self):
        ans = raw_input('do you want to play another game? Y/N')
        if ans == 'Y' or ans == 'y':
            return True
        else:
            return False

    def quit_game(self, player1, player2):
        print('the game has ended, the results are:')
        print(player1.name + ': ' + player1.score, player2.name + ': ' + player2.score)
        if(player1.score > player2.score):
            print("Congrationlations, " + player1.name + '! you won!')
        elif (player2.score > player1.score):
            print("Congrationlations, " + player2.name + '! you won!')
        else:
            print("it's a tie!")

    def play(self):
        player1 = Player()
        player2 = Player()
        player = 1
        board = self.init_board()
        player1_name = raw_input("player 1 - please enter your name: ")
        print("Welcome, " + player1_name + "!")
        player1.name = player1_name
        player2_name = raw_input("player 2 - please enter your name: ")
        print("Welcome, " + player2_name + "!")
        player2.name = player2_name
        res = 0
        stop = False
        while stop is False:
            while res == 0:
                self.move(player, player1_name, player2_name, board)
                res = self.check_board(board)
                player = 3 - player
            self.declare_winner(res, player1_name, player2_name)
            stop = not(self. get_player_request())
        self.quit_game(player1, player2)

    def init_board(self):
        board = [[0 for x in range(self.BOARD_SIZE)] for y in range(self.BOARD_SIZE)]
        return board

    def is_legal(self, move, board):
        splitted = move.split(' ')
        if len(splitted) is not 2:
            return False
        else:
            row = int(splitted[0])
            col = int(splitted[1])
            if row < 0 or row >= self.BOARD_SIZE or col < 0 or col >= self.BOARD_SIZE:
                return False
            else:
                if not (board[row][col] == 0):
                    return False
                else:
                    return True

    def get_sign(self, param):
        if param == 1:
            return 'X'
        elif param == 2:
            return 'O'
        else:
            return ' '

    def print_board(self, board):
        print "the board now is:"
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                # if(i==0):
                # sys.stdout.write("|")
                sys.stdout.write(self.get_sign(board[i][j]) + "|")
            print('')

    def move(self, player, player1_name, player2_name, board):
        if (player == 1):
            name = player1_name
        else:
            name = player2_name
        move = raw_input(
            name + " - please enter your move in the format [ROW] [COL]. where the indices are in the range 0-2: ")
        while not self.is_legal(move, board):
            print "Illegal move"
            move = raw_input(
                player1_name +
                " - please enter your move in the format [ROW][COL]. where the indices are in the range 0-2: ")
        splitted = move.split(' ')
        row = int(splitted[0])
        col = int(splitted[1])
        board[row][col] = player
        self.print_board(board)

    def declare_winner(self, res, player1_name, player2_name):
        # print "res is: ", res
        if res == 3:
            print("it's a tie!")
        else:
            if res == 1:
                name = player1_name
            else:
                name = player2_name
            print("Congratiolations, " + name + "! you won this round!!!")

    def check_rows_or_cols(self, board, r_flag):
        for i in range(self.BOARD_SIZE):
            flag = True
            val = board[i][0] if r_flag == self.ROWS else board[0][i]
            for j in range(1, self.BOARD_SIZE):
                if r_flag == self.COLS:
                    if board[j][i] != val:
                        flag = False
                        break
                else:
                    if board[i][j] != val:
                        flag = False
                        break
            if flag is True:
                return val
        return 0

    def check_main_diagonal(self, board):
        val = board[0][0]
        for i in range(1, self.BOARD_SIZE):
            if board[i][i] != val:
                return 0
        return val

    def check_secondary_diagonal(self, board):
        val = board[self.BOARD_SIZE - 1][self.BOARD_SIZE - 1]
        for i in xrange(self.BOARD_SIZE - 2, -1, -1):
            if board[i][i] != val:
                return 0
        return val

    def check_diagonals(self, board):
        res = self.check_main_diagonal(board)
        if res is not 0:
            return res
        res = self.check_secondary_diagonal(board)
        if res is not 0:
            return res
        else:
            return 0

    def is_full(self, board):
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == 0:
                    return False
        return True

    def check_board(self, board):
        checks = [self.check_rows_or_cols(board, self.ROWS), self.check_rows_or_cols(board, self.COLS),
                  self.check_diagonals(board)]
        if self.is_full(board) and max(checks) == 0:
            return 3
        return max(checks)


def main():
    pass
    game = TicTacToe()
    game.play()

if __name__ == "__main__":
    main()
