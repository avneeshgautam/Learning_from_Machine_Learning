import numpy as np
class Gradient_Descent:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    
    def f(self,w,b,x):
        return 1.0/ ( 1.0 + np.exp(-(w*x + b)) )
    
    def error(self,w,b):
        err = 0.0
        for x,y in zip(self.X, self.Y):
            fx = self.f(w,b,x)
            err += 0.5 + (fx - y) ** 2
        
        return err

    def grad_b(self,w,b,x,y):
        fx = self.f(w,b,x)
        return (fx-y) * fx * (1-fx)

    def grad_w(self, w,b,x,y):
        fx = self.f(w,b,x)
        return (fx-y) * fx * (1-fx) * x
    

    def do_gradient_descent(self):
        w,b,eta,max_epoch = -2, -2 , 1.0, 1000
        for i in range(max_epoch):
            dw,db = 0,0
            for x,y in zip(self.X, self.Y):
                dw += self.grad_w(w,b,x,y)
                db += self.grad_b(w,b,x,y)
                
            w = w - eta*dw
            b = b - eta*db
            print("w -> ",w , " b-> ",b)
    



X = [0.5, 2.5]
Y = [0.2, 0.9]

gd = Gradient_Descent(X,Y)
gd.do_gradient_descent()