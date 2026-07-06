list1 = ['a','b','c']           # string objects
list2 = [1,2,3]                 # integer objects
list3 = [1.5, 3.1, 5.7]         # float objects
                                # however all 3 lists contain the same amount of values 3.

# zipping iterates through the mutiple lists at the same time, in this case 3 lists. They create pulled together in the object "item" below.

for item in zip(list1,list2,list3):
    l1,l2,l3 = item
    print(l1)
    print(l2)
    print(l3)

    input()
    


