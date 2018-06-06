import timeit

def quickSort(arr):
	return quickSortHelper(arr,0,len(arr)-1)
	
def quickSortHelper(arr,start,end):
	if start<end:

		splitpoint = partition(arr,start,end)

		quickSortHelper(arr,start,splitpoint-1)
		quickSortHelper(arr,splitpoint+1,end)
	      
def partition(arr,start,end):
	pivotvalue = arr[start]
	
	leftmark = start+1
	rightmark = end

	done = False
	while not done:
		while leftmark <= rightmark and arr[leftmark] <= pivotvalue:
			leftmark += 1
		while rightmark >= leftmark and arr[rightmark] >= pivotvalue:
			rightmark -= 1
	
		if rightmark < leftmark:
			#found the split point
			done = True
			temp = pivotvalue
			arr[start] = arr[rightmark]
			arr[rightmark] = temp
		else:
			#exchange the values
			temp = arr[leftmark]
			arr[leftmark] = arr[rightmark]
			arr[rightmark] = temp
	
	return rightmark
	
	
arr = input("Insert list of numbers separated with ',':\n")
try:
	arr = arr.split(",")
	arr = [int(num) for num in arr]

except:
	print("Wrong input")
	raise
	
start = timeit.default_timer()
quickSort(arr)
stop = timeit.default_timer()
time = stop-start
print(arr)

print("time: ",time)

input("Press Enter To Exit")
