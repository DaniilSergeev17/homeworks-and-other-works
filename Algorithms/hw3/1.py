n = int(input())
phone_book = dict()
res = []

for i in range(n):
    s = input().split()
    if len(s) == 3:
        num, name = s[1], s[2]
        phone_book[num] = name
    else:
        op, num = s
        if op == 'find':
            res.append(phone_book[num] if num in phone_book else 'not found')
        else:
            if num in phone_book:
                phone_book.pop(num)

print('\n'.join(res))

