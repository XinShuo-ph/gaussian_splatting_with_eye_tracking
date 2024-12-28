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

# std of fovealnet prediction errors in deg
std_layers = [7.487776756286621, 2.257791042327881, 1.0565822124481201, 0.6805307269096375, 0.5396168231964111, 0.5181527137756348]
std_layers = np.array(std_layers)
# paracentral, macular and near peripheral regions in deg (wikipedia, peripheral vision) 
r_cen = 4
r_macular = 9
r_near_peripheral = 30

vision_deg = 120 # field of view = 120 deg
horizontal_resolution = 1920 # horizontal resolution of the display

#convert all deg to pixels
r_cen = r_cen/vision_deg*horizontal_resolution
r_macular = r_macular/vision_deg*horizontal_resolution
r_near_peripheral = r_near_peripheral/vision_deg*horizontal_resolution
std_layers = std_layers/vision_deg*horizontal_resolution

d_error_sigma = 10 # covariance of the error for center of the foveated region
myR = 100

safe_r_cen = []
safe_r_mac = []
safe_r_per = []

for myR in [r_near_peripheral, r_macular, r_cen]:
    plt.figure()
    for d_error_sigma in std_layers:


        mean = np.array([0,0])
        cov = np.array([[d_error_sigma**2,0],[0,d_error_sigma**2]])
        d_samples_2d = np.random.multivariate_normal(mean, cov, 10000)
        d_samples = np.sqrt(d_samples_2d[:,0]**2 + d_samples_2d[:,1]**2)
        if myR > r_macular:
            myr = np.linspace(0.9*myR,myR,51)
        else:
            myr = np.linspace(0,myR,51)
        useful_pix_ratio_list = np.zeros_like(myr)
        unused_pix = np.zeros_like(myr)

        for r in myr:
            myratio =  np.mean(A(r,d_samples, myR))/np.pi/r**2
            print('r = ', r, ' useful_pix_ratio = ', myratio)
            useful_pix_ratio_list[r == myr] = myratio
            unused_pix[r == myr] = (1 - myratio)*np.pi*r**2

        # plot the curve
        plt.plot(myr, useful_pix_ratio_list,label = '$R_{fov}=%.0f$, $\\sigma_{error}$=%.0f' % (myR, d_error_sigma))

        # solve the myr for which useful_pix_ratio = 0.99
        idx = np.argmin(np.abs(useful_pix_ratio_list - 0.99))
        if myR == r_cen:
            safe_r_cen.append(myr[idx])
        if myR == r_macular:
            safe_r_mac.append(myr[idx])
        if myR == r_near_peripheral:
            safe_r_per.append(myr[idx])



    # draw a horizontal line at 0.9
    plt.axhline(y=0.99, color='grey', linestyle='--')
    plt.ylim(0.98,1.001)

    plt.xlabel('r')
    plt.ylabel('useful pixel ratio')
    plt.legend()

    plt.show()
    plt.savefig('useful_pixel_ratio_r%f.png'%myR)

# plot the safe r for each region
plt.figure()
plt.plot(range(1,7), safe_r_cen, label='central')
plt.plot(range(1,7), safe_r_mac, label='near central')
plt.plot(range(1,7), safe_r_per, label='peripheral')
plt.xlabel('Fovealnet layer')
plt.ylabel('rendered region radii')
plt.legend()
plt.show()
plt.savefig('safe_r.png')