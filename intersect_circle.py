# refer to Eq. 14 of https://mathworld.wolfram.com/Circle-CircleIntersection.html
import numpy as np
import matplotlib.pyplot as plt
def A(r,d, R): # normalize everything in units of R, or set R=1
    if r + R <= d:
        return 0
    if d + r <= R:
        return r**2 * np.pi
    if d + R <= r:
        return R**2 * np.pi
    return r**2 * np.arccos((d**2 + r**2 - R**2)/(2*d*r)) + R**2 * np.arccos((d**2 - r**2 + R**2)/(2*d*R)) - 0.5*np.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
# vectorize the function
A = np.vectorize(A)




d_error_sigma = 10 # covariance of the error for center of the foveated region
myR = 100


# for myR in [100,200,300,400,500,600]:
for d_error_sigma in [10,20,30,40,50,60]:


    mean = np.array([0,0])
    cov = np.array([[d_error_sigma,0],[0,d_error_sigma]])
    d_samples_2d = np.random.multivariate_normal(mean, cov, 10000)
    d_samples = np.sqrt(d_samples_2d[:,0]**2 + d_samples_2d[:,1]**2)
    myr = np.linspace(0.9*myR,myR,51)
    useful_pix_ratio_list = np.zeros_like(myr)
    unused_pix = np.zeros_like(myr)

    for r in myr:
        myratio =  np.mean(A(r,d_samples, myR))/np.pi/r**2
        print('r = ', r, ' useful_pix_ratio = ', myratio)
        useful_pix_ratio_list[r == myr] = myratio
        unused_pix[r == myr] = (1 - myratio)*np.pi*r**2

    # plot the curve
    plt.plot(myr, useful_pix_ratio_list,label = '$R_{fov}=%.0f$, $\\sigma_{error}$=%.0f' % (myR, d_error_sigma))



# draw a horizontal line at 0.9
plt.axhline(y=0.99, color='grey', linestyle='--')

plt.xlabel('r')
plt.ylabel('useful pixel ratio')
plt.legend()

plt.show()
plt.savefig('useful_pixel_ratio_r100.png')