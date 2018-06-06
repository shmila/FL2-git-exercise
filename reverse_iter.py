class reverse_iter:
	def __init__(self,list):
		self.i = len(list) - 1
		self.list = list
		
	def __iter__(self):
		return self
	
	def next(self):
		if self.i > -1:
			i = self.list[self.i]
			self.i -=1
			return i
		else:
			raise StopIteration()


it = reverse_iter([1, 2, 3, 4])
for i in range(5):
	print(it.next())