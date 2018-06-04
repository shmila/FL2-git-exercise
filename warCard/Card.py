class Card:
    def __init__(self, card_value, card_sign):
        self.value = card_value
        self.sign = card_sign

    @property
    def value(self):
        return self.value

    @property
    def sign(self):
        return self.sign

    def is_greater_than(self, other):
        return self.value > other.value

    def __str__(self):
        if(self.value == 11):
            val = 'Jack'
        elif(self.value == 12):
            val = 'Queen'
        elif (self.value == 13):
            val = 'King'
        elif (self.value == 1):
            val = 'Ace'
        else:
            val = str(self.value)
        return val + ' of ' + self.sign + 's'
