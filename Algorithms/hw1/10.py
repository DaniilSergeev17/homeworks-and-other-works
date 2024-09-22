n = int(input())

def count_div(n: int) -> set[int]:
    res = set()
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            res.add(i)
            res.add(n // i)
    res.add(n)
    return res

def is_prime_to_each_other(first: int, second: int) -> bool:
    divs_first: set[int] = count_div(first)
    divs_second: set[int] = count_div(second)

    if divs_first.intersection(divs_second) == set():
        return True

    return False

res = []
main_value = 0
for num in range(1, n):
    second_num = n - num
    mini = min(second_num, num)
    maxi = max(second_num, num)
    if is_prime_to_each_other(num, second_num):
        if (mini / maxi) > main_value:
            main_value = mini / maxi
            res = [mini, maxi]

print(*res)