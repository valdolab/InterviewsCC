#oracle

def isPangram(pangram):
    # Write your code here
    #a = 97
    #z = 122
    flag = ""
    for str_p in pangram:
        str_p.lower()
        dic2 = []
        for i in str_p:
            if i not in dic2 and i != " ":
               dic2.append(i)
        if len(dic2) == 26:
            flag = flag + "1"
        else:
            flag = flag + "0"
        
    return flag

asd = ["we promptly judged antique ivory buckles for the next prize","we promptly judged antique ivory buckles for the prizes","the quick brown fox jumps over the lazy dog", "the quick brown fox jump over the lazy dog"]

re = isPangram(asd)

print(re)