from random import shuffle

class card:
	royals = {11:"Prince",12:"Queen",13:"King",14:"Ace"}
	def __init__(self,shape,num):
		self.num = num
		self.shape = shape
		if num == 1:
			self.num = 14
	
	def __repr__(self):
		type = str(self.num)
		if self.num > 10:
			type = self.royals[self.num]
		return type + " of " + self.shape

class deck:
	def __init__(self,cards = None):
		if cards:
			self.cards = cards
		else:
			self.cards = []
			self.shapes = ["diamonds","hearts", "spades", "clubs"]
			for i in range(1,14):
				for s in self.shapes:
					self.cards.append(card(s,i))
			# now shuffle the deck
			shuffle(self.cards)
	def __str__(self):
		st=""
		for card in self.cards:
			st += str(card)+" , "
		st = st[:-3]
		return st
		
	def draw(self):
		if len(self.cards) > 0:
			return self.cards.pop()
		print("Sorry No More Cards Left!")
		return None
	def __len__(self):
		return len(self.cards)
	__repr__ = __str__
		
class player:
	def __init__(self,deck,name):
		self.name = name
		self.d = deck
		self.curr_card = "None"
		self.score = 0
		self.side_deck = []
	def draw(self):
		self.curr_card = self.d.draw()
		if self.curr_card == None and self.side_deck:
			#switch decks
			self.d = deck(shuffle(self.side_deck))
			self.side_deck = []
			self.curr_card = self.d.draw()
		return self.curr_card
	def add_won_cards(self,cards_won):
		for card in cards_won:
			self.side_deck.append(card)
	def add_score(self,score):
		self.score += score	
	def total_cards(self):
		total = 0
		if self.d:
			total += len(self.d)
		if self.side_deck:
			total += len(self.side_deck)
		return total
	def __str__(self):
		return "Player %s has card %s" % (self.name,self.curr_card)
		
class war_game:
	
	def __init__(self,name1,name2):
		self.turn = 0
		self.finish = False
		d = deck()
		split_point = int(len(d.cards)/2)
		self.p1 = player(deck(d.cards[:split_point]),name1)
		self.p2 = player(deck(d.cards[split_point:]),name2)
	
	def compare(c1,c2):
		#return 1 if c1 is higer, 0 if equal or -1 if lower
		if c1.num > c2.num:
			return 1
		elif c1.num < c2.num:
			return -1
		else:
			return 0
	def get_p1(self):
		return self.p1
	def get_p2(self):
		return self.p2
	def play_turn(self):
		if self.finish:
			print("Game is over. Please start new one")
			return
		self.turn += 1
		print("Start round %d:"%self.turn)
		cards_won = []
		res = 0
		while (res == 0):
			c1 = self.p1.draw()
			c2 = self.p2.draw()
			cards_won.append(c1)
			cards_won.append(c2)
			res = war_game.compare(c1,c2)
			if res == 0:
				print("WAR! card is %s" % c1)
				for i in range(3):
					c1 = self.p1.draw()
					c2 = self.p2.draw()
					cards_won.append(c1)
					cards_won.append(c2)
		if res == 1:
			#p1 won this round
			self.p1.add_won_cards(cards_won)
			self.p1.add_score(1)
		else:
			#p2 won this round
			self.p2.add_won_cards(cards_won)
			self.p2.add_score(1)
		
		if p1.total_cards() == 0:
			self.finish = True
			print("Player %s has won this game" % p2.name)
		if p2.total_cards() == 0:
			self.finish = True
			print("Player %s has won this game" % p1.name)

#main
name1 = input("Please insert first player name: \n")
name2 = input("Please insert second player name: \n")
game = war_game(name1,name2)
p1 = game.get_p1()
p2 = game.get_p2()
while (not game.finish):
	ans = input("%s do you want to see the score? y/n " % p1.name)
	if ans == "y":
		print(p1.score)
	ans = input("%s do you want to see the score? y/n " % p2.name)
	if ans == "y":
		print(p2.score)
	ans = input("%s do you want to continue? y/n " % p1.name)
	if ans == "n":
		game.finish = True
		continue
	ans = input("%s do you want to continue? y/n " % p2.name)
	if ans == "n":
		game.finish = True
		continue
	game.play_turn()
print("Game Finished.\n%s score is %d.\n%s score is %d" % (p1.name,p1.score,p2.name,p2.score))			
		
		