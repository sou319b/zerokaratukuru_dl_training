from datetime import date
import pickle  
x = date(2018,1,1)
print(x)

with open("today.plk", "wb") as f:
    pickle.dump(x, f, -1)
    
with open("today.plk", "rb") as f:
    y = pickle.load(f)
    
print(y)
