# ГДЕ-ТО ОШИБКА (говорят, что надо решать "расширенным алгоритмом евклида")

# (a / b) % m = a * b**(-1) % m = a * b**(m-2) % m, где ** -> fast_pow

a, n, m = map(int, input().split())

def nod(n1, n2):
    maxi = max(n1, n2)
    mini = min(n1, n2)
    while maxi % mini != 0:
        mid = maxi % mini
        maxi = mini
        mini = mid
    return mini

def fast_pow(a, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        return fast_pow(a * a % m, n // 2)
    else:
        return a * fast_pow(a, n // 2) % m

res = 0
flag = False
for i in range(1, n+1):
    chislitel = i
    a_i = fast_pow(a, i) % m

    if nod(a_i, m) != 1:
        flag = True
        break

    res = (res + chislitel * fast_pow(a_i, (m-2)) % m) % m


if flag:
    print(-1)
else:
    print(res)