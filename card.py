class Card(object):
    def __init__(self,suit =1, number = 1):
        '''
        :param suit: defult = 1 within 1-4
        :param number: defult = 1 within  1-13
        # defining suits
        # 4 = clubs (tiltan)
        # 3 = spades (Alim)
        # 2 = hearts
        # 1 = diamonds
        '''
        self.number = number
        self.suit = suit

if  __name__=='__main__':
    card = Card(2, number = 12)
    print(card.number)