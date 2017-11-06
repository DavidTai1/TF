def bubbleSort(alist):
    for i in range(len(alist)):
        m = alist[0]
        ind = 0
        for j in range(len(alist)-i):
            if alist[j] < m:
                m = alist[j]
                temp = alist[j-1]
                alist[j-1] = alist[j]
                alist[j] = temp

alist = [-1,100,-2,-200,54,26,93,17,77,31,44,55,20,1000]
bubbleSort(alist)
print(alist)
