Done = False

def dec_val(num , lang):
	counter = 0
	sum = 0
	size = len(lang)
	for dig in reversed(num):
		sum += lang.index(dig)*size**counter
		counter += 1
	return sum
def dest_val(dec_val , dest_lang):
	size = len(dest_lang)
	res = []
	if dec_val == 0:
		return dest_lang[0]
	while (dec_val > 0):
		remainder = dec_val % size
		dec_val = int(dec_val / size)
		remainder = dest_lang[remainder]
		res.insert(0,remainder)
	return res
	

while (not Done):
	case = input("Give me the case: \n")
	if case == "exit":
		Done = True
		continue
	alien_number , source_lang , dest_lang = case.split()
	s_size = len(source_lang)
	d_size = len(dest_lang)
	mid_val = dec_val(alien_number,source_lang)
	dst_val = dest_val(mid_val,dest_lang)
	print(mid_val)
	print(dst_val)
	