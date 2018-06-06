import random

def pass_generator(option):
	#default is easy
	pass_len = 6
	if option == "medium":
		pass_len = 10
	elif option == "strong":
		pass_len = 14

	passW = ""
	for i in range(pass_len):
		if random.random() >= 0.5:
			#generate number
			dig = random.randint(0,9)
			passW += str(dig)
		else:
			#generate char
			ch = random.randint(34,126)
			if ch == 34:
				passW += '!'
			else:
				passW += chr(ch)
	return passW
	
		
cont = "1"
while (cont == "1"):
	pass_strong = input("How much strong do you want your password? easy/medium/strong\n")
	options = {"easy","medium","strong"}
	while (pass_strong not in options):
		pass_strong = input("How much strong do you want your password? easy/medium/strong\n")
	print("Good option was selected : ",pass_strong)
	print("Your password is: ",pass_generator(pass_strong))
	cont = input("Press Enter to exit or 1 to continue\n")
