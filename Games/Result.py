class GameResult:
    def __init__(self, score=0, num_turns = 0,is_tie = False,loseP='',winP=''):
        self.score = score
        self.lose_player_name = loseP
        self.win_player_name = winP
        self.is_tie = is_tie
        self.num_turns = num_turns

    def __str__(self):
        if self.is_tie:
            return "It's a Tie! Number of turns: %d" % self.num_turns
        else:
            return 'Winner: %s , Loser: %s , Score: %d in %d turns' % \
                   (self.win_player_name,self.lose_player_name,self.score,self.num_turns)

class GameHistory:

    def __init__(self,p_name=''):
        self.history = []
        self._player_name = p_name
        self.total_score = 0
        self.num_wins = 0
        self.num_loss = 0
        self.num_ties = 0

    def add_result(self,res):
        self.history.append(res)
        if not res.is_tie:
            if res.win_player_name == self._player_name:
                self.total_score += res.score
                self.num_wins += 1
            if res.lose_player_name == self._player_name:
                self.num_loss += 1
        else:
            # no score added
            self.num_ties += 1

    def remove_res(self,res):
        if res in self.history:
            self.history.remove(res)
            if not res.is_tie:
                if res.win_player_name == self._player_name:
                    self.total_score -= res.score
                    self.num_wins -= 1
                if res.lose_player_name == self._player_name:
                    self.num_loss -= 1
            else:
                self.num_ties -= 1

    def get_num_games(self):
        return self.num_loss+self.num_wins+self.num_ties

    def __str__(self):
        return ('Player %s: Wins: %d , Losses: %d , Ties: %d , Total Score: %d' % \
              (self._player_name,self.num_wins,self.num_loss,self.num_ties,self.total_score))





