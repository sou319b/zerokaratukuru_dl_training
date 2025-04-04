def h(a,b=1,c=1):
    return a * 100 + b * 10 + c

print(h(1,2,3))
print(h(2))
print(h(2,c=2))
print(h(a=2,c=3))
print(h(b=2,a=1,c=3))
