from deck import Deck
from player import Player
# from math import max
from card import Card


class CardWar(object):

    def __init__(self,number_of_players = 2):

        self.players_arr =[]
        for player in range(number_of_players):
            self.players_arr.append(Player())

    def deal(self, shuffle_times = 0 , deck = []):
        if deck == []:
            deck = Deck()
        deck.shuffle(shuffle_times)
        deck.deal(self.players_arr)

    def score(self):
        for player in self.players_arr:
            print("Player " + str(self.players_arr.index(player)) + ", score: " + str(len(player.deck)))

    def playRound(self):
        round_cards = []
        round_nums = {}
        for player in self.players_arr:
            card = player.playCard()
            round_cards.append(card)
            round_nums[player] = card.number
        winner = [player for player in round_nums if
                      round_nums[player] == max(round_nums.values())]  # if player's num == max of round nums add key
        #  (player) to winner index
        print(round_nums)
        print('winner' + str(winner))
        print("num of cards" + str(len(round_cards)))


        while len(winner) > 1:  # check if there was a tie
            round_cards_tie = []
            round_nums_tie = {}
            for player in winner:  ##
                card = player.playCard()
                round_cards_tie.append(card)
                round_nums_tie[player] = card.number
            winner = [player for player in round_nums_tie if
                      round_nums_tie[player] == max(round_nums_tie.values())]  # if player's num == max of round nums add key
        #  (player) to winner list
            round_cards = round_cards + round_cards_tie
            print(round_nums_tie)
            print('winner' + str(winner))
            print("num of cards" + str(len(round_cards)))

        winner[0].recieveCards(round_cards)

#$# add case of player loose, and last player winner
if __name__ == "__main__":
    game = CardWar(2)
    print(game.players_arr)
    deck = Deck(4,13)
    game.deal(100,deck)
    game.score()
    for i in range(10):
        game.playRound()
        game.score()



