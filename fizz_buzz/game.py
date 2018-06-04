import Players
import Result

class FizzGame:

    options = ['Fizz Buzz', 'Fizz', 'Buzz']

    def __init__(self):
        self.p1 = None
        self.p2 = None
        self.num_turns = 0
        self.finished = False
        # players cache
        self.players = {}

    def reset(self):
        self.p1 = None
        self.p2 = None
        self.num_turns = 0
        self.finished = False

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



    def start(self):
        # get the 2 players keys
        keys = self.get_players()
        print('Game is Starting...')
        self.finished = False
        curr_num = 1
        nameP1 = self.p1.name
        nameP2 = self.p2.name
        res = None
        while not self.finished:
            for p in [self.p1,self.p2]:
                p_ans = p.play(curr_num)

                if p_ans != FizzGame.get_correct_answer(curr_num):
                    if p.name == nameP1:
                        win = nameP2
                        lose = nameP1
                    else:
                        win = nameP1
                        lose = nameP2
                    res = Result.GameResult(score=curr_num, num_turns=curr_num, winP=win, loseP=lose)
                    self.finished = True
                    break
                curr_num += 1
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






    @staticmethod
    def get_correct_answer(curr_num):
        if curr_num%3==0 and curr_num%5==0:
            correct = 'Fizz Buzz'
        elif curr_num%3==0:
            correct = 'Fizz'
        elif curr_num%5==0:
            correct = 'Buzz'
        else:
            correct = str(curr_num)
        return correct


if __name__ == "__main__":

    game = FizzGame()
    while True:
        game.start()
        ans = raw_input('One more? Y/N\n').upper()
        if ans == 'N':
            print('Results:')
            for player in game.players.values():
                gh = player.game_history
                print(gh)
            print('bye bye')
            break

