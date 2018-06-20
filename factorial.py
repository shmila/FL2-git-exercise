
num = input("please enter a positive real number\n")

def factorial_func(x):
   max_range=x+1
   count=1
   for i in range(1,max_range):
       count=count*i

   return count

outcome=factorial_func(num)
print("the factorial of the number " + str(num) + " is " + str(outcome))
