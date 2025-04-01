n = int(input())
m = [int(i) for i in input().split()]


dct = dict()

res = 10**6 + 1
for i, num in enumerate(m):
    if num not in dct.keys():
        dct[num] = i
    else:
        res = min(i - dct[num] - 1, res)
        dct[num] = i

if res != 10**6 + 1:
    print(res)
else:
    print(-1)
