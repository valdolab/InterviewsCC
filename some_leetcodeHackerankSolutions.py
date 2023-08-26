#prueba de interview

#=== ejercicio 1
#return a list of n smallest even integers greater than or equal to start,
#in ascending order
def even(start, n):
    # write your code here
    nlist = []
    while(len(nlist) < n):
        if start % 2 == 0:
            nlist.append(start)
        start = start + 1 
        
    return nlist

#print(even(5,7))


#==== otro ejericio
#https://www.geeksforgeeks.org/minimum-number-increment-operations-make-array-elements-equal/
#Given an array of intergers, determine the number of moves to make all elements equal. 
#Each move consists of choosing all but 1 element and incrementing their values by 1
def find_moves(numbers):
    result = numbers.count(numbers[0]) == len(numbers)
    #while(result is not False):
    #    a = 9 
        #y aqui lo puedes hacer con uno chingo de loops, y listas
        #bla bla bla
        
    #PERO la solucion es más aritmetica, y queda así:
    moves = sum(numbers) - (len(numbers) * min(numbers))
    return moves

numbers = [2,2,2]
#print(find_moves(numbers))


#==== problema del triangulo con espacios
def make_tree(n):
    s = ""
    for i in range(n-1):
        s = s + " "
    c = "#"
    for i in range(n):
        print(s + c)
        s = s[0:len(s)-1]
        c = c + "#"

#make_tree(6)


#=== otro ejericicio probando set (hashtable)
#para checar si algun dato existe dentro de una lista en tiempo O(1) y no O(n)

import time 
from decimal import Decimal

a = set(range(10))
b = set(range(1000))
c = set(range(1000000))
d = tuple(range(10000000))

prueba = 9999999
starttime = time.time()
if prueba in a:
    print("Yes")
else:
    print("No")

executionTime = Decimal((time.time() - starttime))
print('Execution time in seconds: ' + str(executionTime))



#EJERCICIOS VIEJOS QUE HACIA CON LIST, AHORA CON SET (O(1) de tiempo)
def findre(arr):
    con = 0
    aux = set()
    for i in arr:
        if i in aux:
            con = con + 1
        else:
            aux.add(i)

    return con

import time
from decimal import Decimal

arr = [2,3,4,2,5,6,7,2,4]
starttime = time.time()
print(findre(arr))

executionTime = Decimal((time.time() - starttime))
print('Execution time in seconds: ' + str(executionTime))












