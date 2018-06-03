def factorialRec(index):
    if index > 1:
        return index*factorialRec(index-1)
    else:
        return 1