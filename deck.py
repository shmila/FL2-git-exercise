from card import Card
from random import randint


class Deck(object):
    def __init__(self, num_suits=4, num_cards=13):
        '''

        :param num_suits: defualt 4
        :param num_cards: in each suits, default 13
        '''
        self.num_suits = num_suits
        self.num_cards = num_cards
        self.deck_arr = []

        for suit in range(1, num_suits + 1):
            for num in range(1, num_cards + 1):
                card = Card(suit, num)
                self.deck_arr.append(card)

    def deal(self, player_arr):
        player_to_deal = 0
        card_arr = ["card"]
        while len(self.deck_arr) > 0:
            card_arr[0] = self.deck_arr[0]  # and array made of the top card
            player_arr[player_to_deal].recieveCards(card_arr)  # give top card to player as a short array
            self.deck_arr.remove(card_arr[0])  # remove card from deck
            player_to_deal = (player_to_deal + 1) % len(player_arr)  # move to next player

    def shuffle(self, numOfShuffles=200):
        for shuffle in range(numOfShuffles):
            idx = randint(0, len(self.deck_arr)-1)  # choose a random card from the deck
            card = self.deck_arr[idx]
            self.deck_arr.remove(card)  # remove card from deck
            self.deck_arr.insert(randint(0, len(self.deck_arr) - 1), card)  # insert card in random place
        print('Deck is shuffled')


if __name__ == '__main__':
    deck = Deck()
    idx = 22
    print('num ' + str(deck.deck_arr[idx].number))
    print('suit ' + str(deck.deck_arr[idx].suit))
    print('card ' + str(deck.deck_arr[idx]))

    # deck.shuffle()
    idx = 18
    print('num ' + str(deck.deck_arr[idx].number))
    print('suit ' + str(deck.deck_arr[idx].suit))
    print('card ' + str(deck.deck_arr[idx]))

    for card in range(len(deck.deck_arr)):
        for equalize in range(card + 1, len(deck.deck_arr)):
            if deck.deck_arr[card].number == deck.deck_arr[equalize].number:
                if deck.deck_arr[card].suit == deck.deck_arr[equalize].suit:
                    print(card)
                    print(equalize)
