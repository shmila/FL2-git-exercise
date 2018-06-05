from Player import Player
from Card import Card
import numpy as np


class WarCard:
    def __init__(self):
        self.player1 = Player()
        self.player2 = Player()
        diamonds = [Card(val, "diamond") for val in range(1, 14)]
        clubs = [Card(val, "club") for val in range(1, 14)]
        hearts = [Card(val, "heart") for val in range(1, 14)]
        spades = [Card(val, "spade") for val in range(1, 14)]
        deck = diamonds+clubs+hearts+spades
        np.random.shuffle(deck)
        self.deck = deck

    def get_deck(self):
        return self.deck

    def play(self):
        player1_name = raw_input("player 1 - please enter your name: ")
        print("Welcome, " + player1_name + "!")
        player2_name = raw_input("player 2 - please enter your name: ")
        print("Welcome, " + player2_name + "!")
        self.player1.name = player1_name
        self.player2.name = player2_name
        playing = True
        while playing is True:
            if len(self.deck) == 0:
                print("deck is empty! game over!")
                break
            show_score = raw_input("do you want to view the score? Y/N: ")
            if (show_score == 'Y ' or show_score == 'y'):
                self.show_score()
            ans = raw_input(player1_name + " - do you want to continue playing? Y/N: ")
            if(ans == 'N ' or ans == 'n'):
                playing = False
                break
            player_1_card = self.deck.pop(0)
            player_2_card = self.deck.pop(0)
            print(self.player1.name + " your card is: " + str(player_1_card))
            print(self.player2.name + " your card is: " + str(player_2_card))
            if(player_1_card.is_greater_than(player_2_card)):
                print(self.player1.name + " your won this round!")
                self.player1.score += 1
            elif(player_2_card.is_greater_than(player_1_card)):
                print(self.player2.name + " your won this round!")
                self.player2.score += 1
            else:
                print("it's a tie")
        self.quit_game()

    def quit_game(self):
        self.show_score()
        print("goodbye!")

    def show_score(self):
        print("the result is: ")
        print(self.player1.name + ": " + str(self.player1.score) + " points")
        print(self.player2.name + ": " + str(self.player2.score) + " points")


war_card = WarCard()
# for card in war_card.get_deck():
#     print("card is: " + str(card))
war_card.play()