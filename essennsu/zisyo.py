d = {"a":1,
     "b":2,
     "c":3}
print(d["a"])
print(d["b"])
print("b" in d
)

d = {}
d["x"] = 1
d["y"] = 2
d["z"] = 3
print(d)
for k in d:
    print(k)

for x in d.items():
    print(x)

a = set()
a.add(1)
a.add(2)
a.add(3)
a.add(3)
print(a)
print(2 in a)
print(5 in a)
b = {2,3,4}
print(b)
print(a & b)
print(a | b)
print(a - b)
for x in a:
    print(x)
    