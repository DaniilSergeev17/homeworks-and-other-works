a = int(input())
b = int(input())

def nod(a, b):
    maxi = max(a, b)
    mini = min(a, b)
    while maxi % mini != 0:
        betw = maxi % mini
        if betw < mini:
            maxi = mini
            mini = betw
        else:
            maxi = betw
    
    return mini

print(nod(a, b))