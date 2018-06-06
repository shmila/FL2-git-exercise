def peep(it):
	cp = []
	for i in it:
		cp.append(i)
	return cp[0] , cp

it = iter(range(5))
x, it1 = peep(it)
print x, list(it1)


	