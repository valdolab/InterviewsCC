#ejemplo de test de prueba oracle

def fizzBuzz(n):
    for i in range(1,n+1):
        c = ''
        if i%3==0:
            c+= 'Fizz'
        if i%5==0:
            c+='Buzz'
        if c=='':
            c=i
        print(c)

if __name__ == '__main__':
    n = int(input().strip())
    fizzBuzz(n)


