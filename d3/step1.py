#step1
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
                
            if x.creator is not None:
                funcs.append(x.creator)
            
        
import numpy as np

# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# x.data = np.array(2.0)
# print(x.data)


#step2
class Function:
    def __call__(self, inputs):
            xs = [x.data for x in inputs]
            ys = self.forward(xs)
            outputs = [Variable(as_array(y)) for y in ys]
            
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = outputs
            return outputs
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx
    
# x = Variable(np.array(10)) # Variableクラスのインスタンスを作成
# f = Square()    # Squareクラス（Functionの子クラス）のインスタンス
# y = f(x)    # ここでfの__call__メソッドが呼ばれる
# print(type(y))
# print(y.data)

#step3
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx
    

# A = Square()
# B = Exp()
# C = Square()
# x = Variable(np.array(0.5))
# a = A(x)
# b = B(a)
# y = C(b)
#print(y.data)

#step4
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

# f = Square()
# x = Variable(np.array(2.0))
# dy = numerical_diff(f, x)
# print(dy)

# def f(x):
#     A = Square()
#     B = Exp()
#     C = Square()
#     return C(B(A(x)))

# x = Variable(np.array(0.5))
# dy = numerical_diff(f, x)
# print(dy)

#step6
# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(x.grad)

#step7
# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a
# assert y.creator.input.creator.input.creator == A
# assert y.creator.input.creator.input.creator.input == x

# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

#step8 逆伝播
# y.grad = np.array(1.0)
# y.backward()
# print(x.grad)

#step9
def square(x):
    return Square()(x)
def exp(x):
    return Exp()(x)

# x = Variable(np.array(0.5))
# y = square(exp(square(x)))
# # y.grad = np.array(1.0)
# y.backward()
# print(x.grad)
# x = np.array(1.0)
# y = x **2
# print(type(x), x.ndim)
# print(type(y))

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

#print(np.isscalar(np.float64(1.0)))


#step10
import unittest
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
    
#unittest.main()

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