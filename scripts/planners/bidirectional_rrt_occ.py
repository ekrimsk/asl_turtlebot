import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments

import time # to allow for timeouts
# Convert RRT Connect to Work on occupnacy grid by combining code from other
# RRTConnect and AStar


# Represents a motion planning problem to be solved using the RRT algorithm
class RRTConnect(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, max_time=0.7):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        
        self.occupancy = occupancy
        self.resolution = resolution

        self.x_init = self.snap_to_grid(np.array(x_init))                  # initial state
        self.x_goal = self.snap_to_grid(np.array(x_goal))                  # goal state
        
        self.path = None        # the final path as a list of states
        self.max_time = max_time
    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRTConnect")

    def find_nearest_forward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering forward from x
        """
        raise NotImplementedError("find_nearest_forward must be overriden by a subclass of RRTConnect")

    def find_nearest_backward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from x to V[i] is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering backward from x
        """
        raise NotImplementedError("find_nearest_backward must be overriden by a subclass of RRTConnect")

    def steer_towards_forward(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRTConnect")

    def steer_towards_backward(self, x1, x2, eps):
        """
        Steers backward from x2 towards x1 along the shortest path (subject
        to robot dynamics). Returns x1 if the length of this shortest path is
        less than eps, otherwise returns the point at distance eps along the
        path backward from x2 to x1.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards_backward must be overriden by a subclass of RRTConnect")

    # COPIED IN FROM AStar 
    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    # Copied in from AStar 
    def get_free_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by simply adding/subtracting self.resolution
               from x, numerical error could creep in over the course of many
               additions and cause grid point equality checks to fail. To remedy
               this, you should make sure that every neighbor is snapped to the
               grid as it is computed.
        """
        neighbors = []
        ########## Code starts here ##########

        # 8 neighbors to check (clockwise from top left) 
        x_offset = [-1, 0, 1, 1,  1,  0, -1, -1]
        y_offset = [ 1, 1, 1, 0, -1, -1, -1,  0]
        for xo, yo in zip(x_offset, y_offset):
            nb = self.snap_to_grid((x[0] + xo*self.resolution, x[1] + yo*self.resolution))
            if self.is_free(nb):
                neighbors.append(nb)
        ########## Code ends here ##########
        return neighbors


    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line.
        """
        ########## Code starts here ##########
        return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)




    def solve(self, eps = 0.1, max_iters = 10000, shortcut=True):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

	print('RRT Trying to solve')
        
        # NOTE: added a default 'eps' may want to change for turtlebot
        timeout = time.time() + self.max_time # 15 seconds from now
        state_dim = len(self.x_init)

        V_fw = np.zeros((max_iters, state_dim))     # Forward tree
        V_bw = np.zeros((max_iters, state_dim))     # Backward tree

        n_fw = 1    # the current size of the forward tree
        n_bw = 1    # the current size of the backward tree

        P_fw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the forward tree
        P_bw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the backward tree

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V_fw, V_bw, P_fw, P_bw, n_fw, n_bw: the represention of the
        #           planning trees
        #    - success: whether or not you've found a solution within max_iters
        #           RRT-Connect iterations
        #    - self.path: if success is True, then must contain list of states
        #           (tree nodes) [x_init, ..., x_goal] such that the global
        #           trajectory made by linking steering trajectories connecting
        #           the states in order is obstacle-free.
        # Hint: Use your implementation of RRT as a reference

        ########## Code starts here ##########

        V_fw[0,:] = self.x_init
        V_bw[0,:] = self.x_goal

        space_diff = self.statespace_hi - self.statespace_lo

        for it in range(max_iters):
            if time.time() > timeout:
                print("RRT TIMED OUT")
                return False
            
            # Forward 
            if not(success):
                tmp_rnd = np.random.rand(state_dim)
                x_rnd = self.statespace_lo + np.multiply(tmp_rnd, space_diff)
                x_rnd = self.snap_to_grid(x_rnd)

                n_par = self.find_nearest_forward(V_fw[0:n_fw, :], x_rnd)
                x_near = V_fw[n_par,:]
                
                x_new = self.steer_towards_forward(x_near, x_rnd, eps)
                x_new = self.snap_to_grid(x_new)

                if self.is_free_motion(x_near, x_new):
                    V_fw[n_fw,:] = x_new
                    P_fw[n_fw] = n_par # add edge
                    n_fw += 1 

                    n_con = self.find_nearest_backward(V_bw[0:n_bw, :], x_new)
                    x_con = V_bw[n_con, :]
                    while True:
                        x_new_con = self.steer_towards_backward(x_new, x_con, eps)
                        x_new_con = self.snap_to_grid(x_new_con)

                        if self.is_free_motion(x_new_con, x_con):
                            V_bw[n_bw, :] = x_new_con
                            P_bw[n_bw] = n_con
                            n_bw += 1 
                            if np.all(x_new_con == x_new):
                                success = True
                                break # will reconstruct pat further down 
                            x_con = x_new_con
                        else:
                            break 

            if not(success):
                # Now backwards 
                tmp_rnd = np.random.rand(state_dim)
                x_rnd = self.statespace_lo + np.multiply(tmp_rnd, space_diff)
                x_rnd = self.snap_to_grid(x_rnd)

                n_par = self.find_nearest_backward(V_bw[0:n_bw, :], x_rnd)
                x_near = V_bw[n_par,:]
                x_new = self.steer_towards_backward(x_rnd, x_near, eps)
                x_new = self.snap_to_grid(x_new)

                if self.is_free_motion(x_new, x_near):
                    V_bw[n_bw,:] = x_new
                    P_bw[n_bw] = n_par # add edge
                    n_bw += 1 

                    n_con = self.find_nearest_forward(V_fw[0:n_fw, :], x_new)
                    x_con = V_fw[n_con, :]
                    while True:
                        x_new_con = self.steer_towards_forward(x_con, x_new, eps)
                        x_new_con = self.snap_to_grid(x_new_con)

                        if self.is_free_motion(x_con, x_new_con):
                            V_fw[n_fw, :] = x_new_con
                            P_fw[n_fw] = n_con
                            n_fw += 1 
                            if np.all(x_new_con == x_new):
                                success = True
                                break # will reconstruct pat further down 
                            x_con = x_new_con
                        else:
                            break 

        if success: # update path 
            # hit break awhen x_new_con = x_new
            # x_new is the most recent node inserted in either tree 

            # so x_new is the connecting node 

            # have path from x_init to x_connect
            # have path from x_connet to x_goal 
            # join them (WITHOUT including x_connect twice)

            path_backward = [x_new]
            n_par = P_bw[n_bw-1]
            while n_par != -1:
                path_backward.append(V_bw[n_par, :])
                n_par = P_bw[n_par]

            # will reverse this one so is starts at x_init 
            path_forward = []  # start with nothing because x_new in the other one 
            n_par = P_fw[n_fw-1]
            while n_par != -1:
                path_forward.append(V_fw[n_par, :])
                n_par = P_fw[n_par]

            path_forward = list(reversed(path_forward))


            self.path = path_forward + path_backward

        ########## Code ends here ##########
 

        if success:
            if shortcut:
                self.shortcut_path()
            print('RRT SOLVED')
        else:
            print('RRT Failure')

        return success


    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        path = self.path # copy the init path 
        success = False 
        while not(success):
            success = True # will get flipped if we update 
            # start at end and move backwards to preserve indexing 
            N = len(path)
            for idx in range(N - 2, 0, -1):
                if self.is_free_motion(path[idx + 1], path[idx - 1]):
                    path.pop(idx)
                    success = False

        self.path = path 

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)



    """Plots the path found in self.path and the obstacles"""
    def plot_path(self, fig_num=0):

        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.array(self.path) * self.resolution
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="RRT Con solution path", zorder=10)
        plt.scatter([self.x_init[0]*self.resolution, self.x_goal[0]*self.resolution], [self.x_init[1]*self.resolution, self.x_goal[1]*self.resolution], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", np.array(self.x_init)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

class GeometricRRTConnect(RRTConnect):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return np.argmin(np.linalg.norm(V - x, axis=1))  # copied from P2 
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        
        # copied from P2
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)

        dist = self.distance(x1, x2)  # because tuples 
        if dist < eps:
            result = x2
        else:
            result =  x1 + (eps/dist)*(x2 - x1)

        return tuple(result)     
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, x1, x2):
        # Check against the occupancy grid 
        #dist = np.linalg.norm(x2 - x1)
        dist = self.distance(x1, x2)  # because tuples 
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)


        # Base Cases  
        if not self.occupancy.is_free(x1) or not self.occupancy.is_free(x2):
            return False 
        if self.occupancy.is_free(x1) and self.occupancy.is_free(x2)    \
                                          and dist < self.resolution:
            return True 
        else:  # else recursion
            x_mid = 0.5*(x1 + x2)
            
            start_is_free = self.is_free_motion(x1, x_mid)
            end_is_free = self.is_free_motion(x_mid, x2)
            return (start_is_free and end_is_free)


        """
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True
        """





    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_tree_backward(self, V, P, **kwargs):
        self.plot_tree(V, P, **kwargs)

    #def plot_path(self, **kwargs):
    #    path = np.array(self.path)
    #    plt.plot(path[:,0], path[:,1], **kwargs)




## Fully copied from AStar -- could move to own class 

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles"""
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True



    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
