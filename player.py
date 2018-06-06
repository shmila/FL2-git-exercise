import random
from card import Card
from deck import Deck


class Player(object):
    '''

    '''
    def __init__(self,name = "Dan", deck = []):
        self.name = name
        self.deck = deck

    def recieveCards(self,card_arr):
        self.deck = self.deck + card_arr
        # for card in card_arr:
        #     self.deck.append(card)

    def playCard(self):
        # if len(self.deck) == 0:
        #     except('Player is out of cards --> GAME OVER')
        card = self.deck[0]  # type: card object to be return
        del self.deck[0] # remove card from deck
        return card

if __name__ == '__main__':
    deck = Deck()
    deck.shuffle(100)
    d = Player()
    d.recieveCards(deck.deck_arr)
    card = d.playCard()
    print('num = ' + str(card.number))
