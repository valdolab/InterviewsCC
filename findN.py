#dada una cadena de n caracteres más grande que p y q, por ejemplo: "abcabc"
#1. mover p caracteres del final al inicio, por ejemplo: p=1
#2. mover q caracteres del final al inicio, por ejemplo: q=1
#3. Y detenerce cuando se llegue a la cadeba original
#4. retornar el numero de pasos que se necesitaron para llegar a la cadena orignal
#
from collections import deque

def findN(string,p,q):
    niter = 0
    #listStr = [i for i in string]
    listStr = deque([i for i in string])
    while True:
        flag = False
        for i in range(p):
            listStr.appendleft(listStr.pop())
            niter = niter + 1
            st = ''.join(listStr)
            if st == string:
                flag = True
        
        if flag:
            break
        
        for i in range(q):
            listStr.appendleft(listStr.pop())
            niter = niter + 1
            st = ''.join(listStr)
            if st == string:
                flag = True
        if flag:
            break
            
    return niter


string = "abcabc"
p = 1
q = 1
print(findN(string,p,q))

