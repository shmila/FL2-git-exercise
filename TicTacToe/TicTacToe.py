import sys

from scipy.signal.ltisys import bode

BOARD_SIZE = 3
ROWS = 1
COLS = 0


def tic_tac_toe():
    player = 1
    board = init_board()
    player1_name = raw_input("player 1 - please enter your name: ")
    print("Welcome, " + player1_name + "!")
    player2_name = raw_input("player 2 - please enter your name: ")
    print("Welcome, " + player2_name + "!")
    res = 0
    while res == 0:
        move(player, player1_name, player2_name, board)
        res = check_board(board)
        player = 3 - player
    declare_winner(res, player1_name, player2_name)


def init_board():
    board = [[0 for x in range(BOARD_SIZE)] for y in range(BOARD_SIZE)]
    return board


def is_legal(move, board):
    splitted = move.split(' ')
    if len(splitted) is not 2:
        return False
    else:
        row = int(splitted[0])
        col = int(splitted[1])
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return False
        else:
            if not (board[row][col] == 0):
                return False
            else:
                return True


def get_sign(param):
    if param == 1:
        return 'X'
    elif param == 2:
        return 'O'
    else:
        return ' '


def print_board(board):
    print "the board now is:"
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            # if(i==0):
            # sys.stdout.write("|")
            sys.stdout.write(get_sign(board[i][j]) + "|")
        print('')


def move(player, player1_name, player2_name, board):
    if (player == 1):
        name = player1_name
    else:
        name = player2_name
    move = raw_input(
        name + " - please enter your move in the format [ROW] [COL]. where the indices are in the range 0-2: ")
    while not is_legal(move, board):
        print "Illegal move"
        move = raw_input(
            player1_name+" - please enter your move in the format [ROW][COL]. where the indices are in the range 0-2: ")
    splitted = move.split(' ')
    row = int(splitted[0])
    col = int(splitted[1])
    board[row][col] = player
    print_board(board)


def declare_winner(res, player1_name, player2_name):
    if res == 3:
        print("it's a tie!")
    else:
        if res == 1:
            name = player1_name
        else:
            name = player2_name
        print("Congratiolations, " + name + "! you won!!!")


def check_rows_or_cols(board, r_flag=ROWS):
    for i in range(BOARD_SIZE):
        flag = True
        val = board[i][0] if r_flag == ROWS else board[0][i]
        for j in range(1, BOARD_SIZE):
            if not r_flag:
                i, j = j, i
            if board[i][j] != val:
                flag = False
                break
        if flag is True:
            return val
    return 0


def check_main_diagonal(board):
    val = board[0][0]
    for i in range(1, BOARD_SIZE):
        if board[i][i] != val:
            return 0
    return val


def check_secondary_diagonal(board):
    val = board[BOARD_SIZE - 1][BOARD_SIZE - 1]
    for i in xrange(BOARD_SIZE - 2, -1, -1):
        if board[i][i] != val:
            return 0
    return val


def check_diagonals(board):
    res = check_main_diagonal(board)
    if res is not 0:
        return res
    res = check_secondary_diagonal(board)
    if res is not 0:
        return res
    else:
        return 0


def is_full(board):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                return False
    return True


def check_board(board):
    checks = [check_rows_or_cols(board, ROWS), check_rows_or_cols(board, COLS), check_diagonals(board)]
    if is_full(board) and max(checks) == 0:
        return 3
    return max(checks)


def main():
    tic_tac_toe()


if __name__ == "__main__":
    main()
