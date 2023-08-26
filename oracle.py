#oracle

cases = "3\nhulk\nmounthulk\nteddy\ncome\ncoder\nurredoc"
#cases =	"1\naaaaaaa\nhaiagahafajabaa"
tokens = cases.split()
num_cases = int(tokens[0])

outputs = []
c=1
dicc = []
for i in range(num_cases):
    pattern = list(tokens[i+c])
    text_str = list(tokens[i+c+1])
    dicc = (pattern+text_str)
    dicc = list(set(dicc))
    v1 = []
    v2 = []
    #for j in range(len(dicc)):         
    for j in range(len(dicc)):
        v1.append(pattern.count(dicc[j]))
        v2.append(text_str.count(dicc[j]))
    flag = True
    for j in range(len(v1)):
        if v1[j]>v2[j]:
            flag = False
            break
    if flag==True:
        outputs.append("Yes")
    else:
        outputs.append("No")
    c += 1
print(outputs)


