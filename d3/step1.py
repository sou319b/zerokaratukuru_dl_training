import numpy as np
import weakref
import contextlib

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

#step1
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
            
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            #gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
                    
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)
                
            # if x.creator is not None:
            #     funcs.append(x.creator)
    
    def cleargrad(self):
        self.grad = None
            
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def __mul__(self, other):
        return mul(self, other)
    

        


# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# x.data = np.array(2.0)
# print(x.data)

class Config:
    enable_backprop = True
    


#step2
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
            
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
                
        return outputs if len(outputs) > 1 else outputs[0]
    
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

#step11
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy

# xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# print(y.data)

#step12
def add(x0, x1):
    return Add()(x0, x1)

# x0 = Variable(np.array(2))
# x1 = Variable(np.array(3))
# y = add(x0, x1)
# print(y.data)


#step13
class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx

# x = Variable(np.array(2.0))
# y = Variable(np.array(3.0))
# z = add(square(x), square(y))
# z.backward()
# print(z.data)
# print(x.grad)
# print(y.grad)


#step14
# x = Variable(np.array(3.0))
# y = add(x, x)
# y.backward()
# print(x.grad)

# x.cleargrad()
# y = add(add(x, x), x)
# y.backward()
# print(x.grad)

#step16
# x = Variable(np.array(2.0))
# a = square(x)
# y = add(square(a), square(a))
# y.backward()
# print(y.data)
# print(x.grad)

#step17
# for i in range(10):
#     x = Variable(np.random.randn(10000))
#     y = square(square(square(x)))
    
#step18
# x0 = Variable(np.array(1.0))
# x1 = Variable(np.array(1.0))
# t = add(x0, x1)
# y = add(x0, t)
# y.backward()
# print(y.grad, t.grad)
# print(x0.grad, x1.grad)




    
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
        
# with using_config('enable_backprop', False):
#     x = Variable(np.array(2.0))
#     y = square(x)
def no_grad():
    return using_config('enable_backprop', False)
# with no_grad():
#     x = Variable(np.array(2.0))
#     y = square(x)
    

#step19
# x = Variable(np.array([1,2,3]))
# print(x)

#step20
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    return Mul()(x0, x1)


# Variable.__mul__ = mul
# Variable.__add__ = add
# a = Variable(np.array(3.0))
# b = Variable(np.array(2.0))
# c = Variable(np.array(1.0))
# y = a * b + c
# y.backward()
# print(y)
# print(a.grad)
# print(b.grad)

#step21
# def as_variable(obj):
#     if isinstance(obj, Variable):
#         return obj
#     return Variable(obj)

x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)
