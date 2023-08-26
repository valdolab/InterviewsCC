#prueba hackerrank and leetcode

#1. ejercicio de sumar las diagonales de una matriz cuadrada, y luego devolver
#la resta absoluta de esas sumas
def diagonalDifference(arr):
    fd = 0
    sd = 0
    sizecube = len(arr)
    for i in range(sizecube):
        fd = fd + arr[i][i]
        sd = sd + arr[i][sizecube-1-i]
    
    return abs(fd - sd)

arr = [[11, 2, 4, 3], [4, 5, 6, 1], [10, 8, -12, 3], [10, 8, -12, 2]]
re = diagonalDifference(arr)

#2. ejercicio


