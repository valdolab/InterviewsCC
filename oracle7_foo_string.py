
lista = ['foo','foobar', 'foospam']    
#sacar el string minimo(menor caracteres), para usarlo como referencia en la busqueda
minword = min(lista,key=len)
#sacar indice del minimo para omitirlo en la busqueda
i_minword = lista.index(minword)
tprefix = ""
for i in range(len(minword)):
    tprefix +=  minword[i]
    for j in range(len(lista)):
        #el primer if es para saltarse la palabra minword, la de referencia
        #podria omitirse, pero haria operaciones de más
        if j != i_minword: 
            if tprefix != lista[j][0:i+1]:
                break
    else:
        prefix = tprefix
        continue
    break

print("\nprefijo: ", prefix)

