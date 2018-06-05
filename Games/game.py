import Players
import Result

class Game(object):

    RESULTS = {"win":1,"lose":-1,"tie":0}

    def __init__(self):
        self.p1 = None
        self.p2 = None
        self.num_turns = 1
        self.finished = False
        # players cache
        self.players = {}

    def reset(self):
        self.p1 = None
        self.p2 = None
        self.num_turns = 1
        self.finished = False

    def get_players(self):
        """
        initial the self.p1 and self.p2 members.
        update the players in self.player
        return their keys
        :return: the 2 players keys
        """
        raise NotImplementedError

    def start(self):
        # get the 2 players keys
        keys = self.get_players()
        print('Game is Starting...')
        self.finished = False
        nameP1 = self.p1.name
        nameP2 = self.p2.name
        res = None
        while not self.finished:
            curr_players = [self.p1, self.p2]
            for p in curr_players:
                # play logic
                self.finished, res_ind, score = self.play_turn(p)
                # check if game is finished
                if self.finished:

                    other_name = list(set([pl.name for pl in curr_players]) - {p.name})[0]

                    if res_ind == self.RESULTS["tie"]:
                        res = Result.GameResult(score=score, num_turns=self.num_turns,is_tie=True)
                        break
                    elif res_ind == self.RESULTS["win"]:
                        win = p.name
                        lose = other_name
                    else:
                        win = other_name
                        lose = p.name
                    res = Result.GameResult(score=score, num_turns=self.num_turns,is_tie=False, winP=win, loseP=lose)
                    break

                self.num_turns += 1
        print('Game Finished')
        print(res)
        # update result in 2 player
        self.p1.game_history.add_result(res)
        self.p2.game_history.add_result(res)
        # update in cache
        if len(keys) == 2:
            self.players[keys[0]] = self.p1
            self.players[keys[1]] = self.p2
        else:
            raise Exception('Error in calculating the players keys')
        self.reset()

    def play_turn(self,p):
        """
        play 1 turn
        :param p: the player object
        :return: tuple: boolean if the game is finished, result of game according to player, winner score
        """
        raise NotImplementedError

    def print_results(self):
        print('Results:')
        for player in game.players.values():
            gh = player.game_history
            print(gh)




class FizzGame(Game):

    options = ['Fizz Buzz', 'Fizz', 'Buzz']


    def __init__(self):
        super(FizzGame,self).__init__()


    def get_players(self):
        keys = []
        for i in range(2):
            p_type = raw_input("Please choose player %d type: H (human) or C (computer)\n" % (i+1)).upper()

            name = raw_input('Player name:\n')

            # calculate  the key
            key = p_type + name
            keys.append(key)
            if key in self.players:
                p = self.players[key]
            else:
                if p_type == 'C':
                    level = input('Please choose computer player %d level: 1 - 3\n' % (i+1))
                    p = Players.ComputerPlayer(name, level)
                else:
                    p = Players.HumanPlayer(name)

                self.players[key] = p
            # check which player are we at
            if i == 0:
                # player 1
                self.p1 = p
            else:
                self.p2 = p
        return keys

    def play_turn(self,p):

        p_ans = p.play(self.num_turns).title()
        if p_ans != FizzGame.get_correct_answer(self.num_turns):
            return (True,self.RESULTS["lose"],self.num_turns)
        else:
            return (False,self.RESULTS["lose"],self.num_turns)


    @staticmethod
    def get_correct_answer(curr_num):
        if curr_num%3==0 and curr_num%5==0:
            correct = FizzGame.options[0]
        elif curr_num%3==0:
            correct = FizzGame.options[1]
        elif curr_num%5==0:
            correct = FizzGame.options[2]
        else:
            correct = str(curr_num)
        return correct


class TicTacToeGame(Game):

    ROWS = 1
    COLS = 0

    def __init__(self):
        super(TicTacToeGame,self).__init__()
        self.board_size = 3
        self.board = None
        self.player = 1
        self.res = 0

    def reset(self):
        super(TicTacToeGame,self).reset()
        self.player = 1
        self.res = 0

    def initBoard(self):
        board = [[0 for x in range(self.board_size)] for y in range(self.board_size)]
        return board

    def get_players(self):
        keys = []
        self.board_size = int(raw_input("Enter board size: \n"))
        self.board = self.initBoard()
        player1_name = raw_input("player 1 - please enter your name: ")
        print("Welcome, " + player1_name + "!")
        player2_name = raw_input("player 2 - please enter your name: ")
        print("Welcome, " + player2_name + "!")

        keys.append(player1_name)
        keys.append(player2_name)

        if player1_name in self.players:
            self.p1 = self.players[player1_name]
        else:
            p = Players.TicTacPlayer(player1_name)
            self.p1 = p
            self.players[player1_name] = p
        if player2_name in self.players:
            self.p2 = self.players[player2_name]
        else:
            p = Players.TicTacPlayer(player2_name)
            self.p2 = p
            self.players[player2_name] = p
        return keys

    def play_turn(self,p):
        p.play(self.player,self.board)
        self.player = 3 - self.player
        res = self.checkBoard()
        score = 1
        if res != 0:
            if res == 3:
                return True, self.RESULTS["tie"], score
            if p.name == self.p1.name:
                # current player is player 1
                if res == 1:
                    return True,self.RESULTS["win"],score
                else:
                    return True,self.RESULTS["lose"], score
            else:
                #current player is player 2
                if res == 2:
                    return True,self.RESULTS["win"], score
                else:
                    return True,self.RESULTS["lose"],score
        else:
            return False,None,score


    # all the checkings!!!
    def checkBoard(self):
        checks = [self.checkRowsOrCols(self.ROWS), self.checkRowsOrCols(self.COLS), self.checkDiagonals()]
        if self.isFull() and max(checks) == 0:
            return 3
        return max(checks)

    def checkSecondaryDiagonal(self):
        val = self.board[self.board_size - 1][self.board_size - 1]
        for i in xrange(self.board_size - 2, -1, -1):
            if self.board[i][i] != val:
                return 0
        return val

    def checkDiagonals(self):
        res = self.checkMainDiagonal()
        if res is not 0:
            return res
        res = self.checkSecondaryDiagonal()
        if res is not 0:
            return res
        else:
            return 0

    def checkRowsOrCols(self, r_flag=1):
        for i in range(self.board_size):
            flag = True
            val = self.board[i][0] if r_flag == self.ROWS else self.board[0][i]
            for j in range(1, self.board_size):
                if not r_flag: i, j = j, i
                if self.board[i][j] != val:
                    flag = False
                    break
            if flag is True:
                return val
        return 0

    def checkMainDiagonal(self):
        val = self.board[0][0]
        for i in range(1, self.board_size):
            if self.board[i][i] != val:
                return 0
        return val

    def isFull(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    return False
        return True




if __name__ == "__main__":
    game = TicTacToeGame()
    while True:
        game.start()
        ans = raw_input('One more? Y/N\n').upper()
        if ans == 'N':
            game.print_results()
            print('Bye Bye')
            break

