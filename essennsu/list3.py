l = [1,2,3]
m = l
m.append(4)
print(m)
print(l)
l = [1,2,3]
m = l[:]
m.append(4)
print(m)
print(l)

l = [[],[],[]]
l[1].append(1)
print(l)
l[2].append(2)
l[2].append(3)
print(l)
 
l = [i**2 for i in range(5)]
print(l)
m = [[i * 10 + j for j in range(5)] for i in range(5)]
print(m)

t = 1,"a",1.5
print(t)

u = t, (1,2,3)
print(u)
print(t[1])
#t[1] = 1

t = ()
print(t)
u = 1,
print(u)

for i in range(5):
    print(i)

l = [2,4,6]
for x in l:
    print(x)

m = [x * 2 for x in l]
print(m)

s = "abcd"
for x in s:
    print(x)

print( 
["*" + x + "*" for x in s]
)

print(list(s))
