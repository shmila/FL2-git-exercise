import Result
import random
from game import FizzGame


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