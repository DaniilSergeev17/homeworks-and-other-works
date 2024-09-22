m, n = map(int, input().split())

eratosfen = [1] * (n+1) # сумма делителей для каждого числа # +1 для нуля 
for i in range(2, n+1):
    for j in range(2*i, n+1, i):
        eratosfen[j] += i

result = []
for i in range(m, n + 1):
    summ_del = eratosfen[i]
    if summ_del > i and summ_del <= n:  
        if eratosfen[summ_del] == i:
            result.append((i, summ_del))            

if len(result) == 0:
    print('Absent')
else:
    for res in result: 
        print(*res)
