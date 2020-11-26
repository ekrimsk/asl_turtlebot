import numpy as np
import scipy.interpolate
from scipy.interpolate import interp1d

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########

    path = np.array(path)

    x = path[:, 0]
    y = path[:, 1]
    N = path.shape[0]

    if N < 8: 
        print("path lengh too short, resampling")
        
        pts = np.array(range(N))
        N_new = 20 
        fx = interp1d(pts, x)
        fy = interp1d(pts, y)
        new_pts = np.linspace(0, N - 1, num=N_new)
        x =  fx(new_pts)
        y = fy(new_pts)
        N = N_new

    diff_x = x[1:] - x[0:-1]
    diff_y = y[1:] - y[0:-1]

    # use to calculate initial time points 
    diff_dist = np.sqrt(diff_x**2 + diff_y**2)
    t_init = np.zeros(N)
    t_init[1:] = np.cumsum(diff_dist/V_des);

    x_spl = scipy.interpolate.splrep(t_init, x, s=alpha)
    y_spl = scipy.interpolate.splrep(t_init, y, s=alpha)

    dx_spl = scipy.interpolate.splder(x_spl)
    dy_spl = scipy.interpolate.splder(y_spl)
    ddx_spl = scipy.interpolate.splder(dx_spl)
    ddy_spl = scipy.interpolate.splder(dy_spl)

    t_smoothed = np.arange(t_init[0], t_init[-1]+dt, dt)

    xs = scipy.interpolate.splev(t_smoothed, x_spl)
    ys = scipy.interpolate.splev(t_smoothed, y_spl)
    dxs = scipy.interpolate.splev(t_smoothed, dx_spl)
    dys = scipy.interpolate.splev(t_smoothed, dy_spl)
    ddxs = scipy.interpolate.splev(t_smoothed, ddx_spl)
    ddys = scipy.interpolate.splev(t_smoothed, ddy_spl)

    ths = np.arctan2(dys, dxs)   # from the dynamics 

    traj_smoothed = np.stack((xs, ys, ths, dxs, dys, ddxs, ddys), axis=1)

    # EREZ ADDED 
    path_length = 0
    for i in range(len(xs) - 1):
        segment_length = np.sqrt((xs[i + 1] - xs[i])**2 + (ys[i + 1] - ys[i])**2)
        path_length = path_length + segment_length

 
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed, path_length
