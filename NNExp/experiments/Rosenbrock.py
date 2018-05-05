
# coding: utf-8

# In[2]:

import matplotlib.pylab as plt
import numpy as np
np.seterr(all='ignore')


# Consider the Rosenbrock function
# $$f(x,y)=10(y-x^2)^2 + (1-x)^2$$
# with gradient
# $$\nabla f = \left[\begin{array}{c}
# 40x^3 - 40xy +2x - 2 \\\
# 20(y-x^2)
# \end{array}\right]$$
# and Hessian
# $$\nabla^2 f = \left[
# \begin{array}{c}
# 120x^2-40y+2 & -40x \\\
# -40x & 20
# \end{array}\right]$$
# The only minimum is at $(x,y)=(1,1)$ where $f(1,1)=0$.

# In[3]:

def objfun(x,y):
    return 10*(y-x**2)**2 + (1-x)**2
def gradient(x,y):
    return np.array([-40.0*x*y + 40*x**3 -2 + 2*x , 20.0*(y-x**2)])
def hessian(x,y):
    return np.array([[120*x*x - 40*y+2, -40*x],[-40*x, 20]])


# Create a utility function that plots the contours of the Rosenbrock function.

# In[4]:

def contourplot(objfun, xmin, xmax, ymin, ymax, ncontours=50, fill=True):
    x = np.linspace(xmin, xmax, 200)
    y = np.linspace(ymin, ymax, 200)

    X, Y = np.meshgrid(x,y)
    Z = objfun(X,Y)
    if fill:
        plt.contourf(X,Y,Z,ncontours); # plot the contours
    else:
        plt.contour(X,Y,Z,ncontours); # plot the contours
    plt.scatter(1,1,marker="x",s=50,color="r");  # mark the minimum


# Here is a contour plot of the Rosenbrock function, with the global minimum marked with a red cross.

# In[5]:

contourplot(objfun, -7,7, -10, 40, fill=False)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of $f(x,y)=10(y-x^2)^2 + (1-x)^2$");


# # Steepest descent (gradient descent) with fixed step length

# First we write a function that uses the steepest descent method. Starts the solution at position `init`, moves along the opposite of the gradient with steps `steplength`, until the absolute difference between function values drops below `tolerance` or until the number of iterations exceeds `maxiter`.
# 
# The function returns the array of all intermediate positions, and the array of function values.

# In[6]:

def steepest_descent(objfun, gradient, init, tolerance=1e-6, maxiter=10000, steplength=0.01):
    p = init
    iterno=0
    parray = [p]
    fprev = objfun(p[0],p[1])
    farray = [fprev]
    while iterno < maxiter:
        p = p - steplength*gradient(p[0],p[1])
        fcur = objfun(p[0], p[1])
        if np.isnan(fcur):
            break
        parray.append(p)
        farray.append(fcur)
        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        iterno += 1
    return np.array(parray), np.array(farray)


# In[14]:

def inv_descent(objfun, gradient, init, tolerance=1e-6, maxiter=3000, steplength=0.01):
    def clip_val(p):
        return np.fabs(p)
    p = init
    iterno=0
    parray = [p]
    fprev = objfun(p[0],p[1])
    farray = [fprev]
    while iterno < maxiter:
        g = gradient(p[0],p[1])
        g[g == 0] = np.inf
        c = clip_val(p)
        delta = -0.01 * fprev / g
        delta_clipped = np.clip(delta, -c, c)
        p_new = p + delta_clipped
        fcur = objfun(p_new[0], p_new[1])
        if fcur - fprev > 0.5:
            print('no')
        p = p_new
        if np.isnan(fcur):
            break
        parray.append(p)
        farray.append(fcur)
        if abs(fcur-fprev)<tolerance:
            break
        fprev = fcur
        iterno += 1
    return np.array(parray), np.array(farray)


# Here's the function to plot the return value:

# In[8]:

def plot_descent(p, f, xmin, xmax, ymin, ymax, iterations):
    plt.figure(figsize=(17,5))
    plt.subplot(1,2,1)
    contourplot(objfun, xmin, xmax, ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Minimize $f(x,y)=10(y-x^2)^2 + (1-x)^2$");
    plt.scatter(p[0,0],p[0,1],marker="*",color="w")
    for i in range(1,len(p)):    
            plt.plot( (p[i-1,0],p[i,0]), (p[i-1,1],p[i,1]) , "w");
    plt.subplot(1,2,2)
    plt.plot(f)
    plt.xlim(0,iterations)
    plt.xlabel("iterations")
    plt.ylabel("function value");


# Now let's see how the steepest descent method behaves with the Rosenbrock function.

# ## Case 1

# Plot the convergence of the solution. Left: The solution points (white) superposed on the contour plot. The star indicates the initial point. Right: The objective function value at each iteration.

# In[9]:

#p, f = steepest_descent(objfun, gradient, init=[2,4], steplength=0.005)
#plot_descent(p,f,-1,3,0,10, 3000)


# In[15]:

p, f = inv_descent(objfun, gradient, init=[2,4], steplength=0.005)
plot_descent(p,f,-1,3,0,10, 30)
p, f


# This solution happens to be too neat. Suppose we increase the step size from $\alpha=0.005$ to $\alpha=0.01$, and the trajectory of the solution gets weird.

# In[11]:

p, f = steepestdescent(objfun, gradient, init=[2,4], steplength=0.01)
plot_descent(p,f, -2,3,0,10, 1000)


# Now the step size is larger, so the new position ends up in a location where the gradient is larger. Therefore the next step is even larger, and we observe a large jump across the middle hill. There the steps get smaller again, and the solution approaches the global minimum from the back.
# 
# Try a different starting location, where the gradient is larger. Now, the same $\alpha$ is too large; the step size increases at each iteration and the calculation blows up.

# In[ ]:

p, f = steepestdescent(objfun, gradient, init=[2,6], steplength=0.01)


# We see that the function value increases rapidly at each iteration. The algorithm is unstable.

# In[ ]:

f


# However, when $\alpha$ is decreased to $0.005$ again, we see that the solution converges after some oscillations.

# In[ ]:

p, f = steepestdescent(objfun, gradient, init=[2,6], steplength=0.005)
plot_descent(p,f, -2,3,0,10, 100)


# In general, the convergence depends sensitively on the $\alpha$ value as well as the local gradient value at the initial position. You can play with various initial positions and step lengths to see how it works.
# 
# Tip: By trial and error, try to find parameter values that are very close to being unstable - they give you crazy paths. For example: 

# In[ ]:

p, f = steepestdescent(objfun, gradient, init=[2,5.1155], steplength=0.01)
plot_descent(p,f, -3,3,0,10,1000)


# In[ ]:



