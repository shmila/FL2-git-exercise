one_txt = ""
other_txt = ""

with open('one.txt', 'r') as open_file:
	one_txt = open_file.read()
with open('other.txt', 'r') as open_file:
	other_txt = open_file.read()
	
listA = one_txt.split()
listB = other_txt.split()	
dif = set(listA) - set(listB)
dif = sorted(list(map(int, dif)))
print(dif)
input("Press Enter To Exit")