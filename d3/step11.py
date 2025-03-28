from step1 import Variable, as_array
import numpy as np

#step11
class Function:
        def __call__(self, input):
            xs = [x.data for x in input]
            ys = self.forward(xs)
            outputs = [Variable(as_array(y)) for y in ys]
            for output in outputs:
                output.set_creator(self)
        
            self.input = input
            self.output = output
            return output
    
        def forward(self, x):
            raise NotImplementedError()
        
        def backward(self, gy):
            raise NotImplementedError()
        
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
