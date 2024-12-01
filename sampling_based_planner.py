import numpy as np
import random
import math


class RRTNode:
    def __init__(self, x, y, theta, vl, vr, arc_rad, v_comm, parent=None):
        self.x = x
        self.y = y
        self.theta = theta          # Heading Angle
        self.vl = vl                # Left Wheel Velocity
        self.vr = vr                # Right Wheel Velocity
        self.arc_rad = arc_rad      # Arc Radius of Motion
        self.v_comm = v_comm        # Translational Velocity
        self.parent = parent        # Parent Node


class RRTConnectPlanner:
    def __init__(self, start_x, start_y, goal_x, goal_y, other_args):
        '''
        Initialization of RRT-Connect planner with robot dynamics
        start_x, start_y: Start position
        goal_x, goal_y: Goal position
        dt: Time step
        rad: Wheel radius
        width: Axle width
        wheelbase: Length between front and rear axles
        goal_radius: Radius within which the goal is considered reached
        max_samples: Maximum number of iterations/samples to attempt
        '''
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.dt = other_args[0]
        self.rad = other_args[1]
        self.width = other_args[2]
        self.wheelbase = other_args[3]
        self.goal_radius = other_args[4]
        self.max_samples = other_args[5]

        self.tree_start = []  # Tree growing from start
        self.tree_goal = []   # Tree growing from goal

    def distance(self, x1, y1, x2, y2):
        '''
        Euclidean Distance
        '''
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def velocity_control(self, rad_comm, velo_comm, th_prev):
        '''
        Computes left and right wheel velocities
        '''
        th_comm = np.arctan(self.wheelbase / (2 * rad_comm))
        ang_velo = velo_comm / rad_comm
        phys_term = np.array([[1 / np.cos(th_comm), -self.width / 2], 
                              [1 / np.cos(th_comm), self.width / 2]]) @ np.array([[velo_comm, ang_velo]]).T
        gain_term = 0  # No gain used in this example
        velos = phys_term + gain_term
        return velos

    def random_sample(self):
        '''
        Generate a random sample in the workspace
        '''
        x = random.uniform(0, 5)  # Workspace dimensions
        y = random.uniform(0, 5)
        theta = random.uniform(0, 2 * np.pi)
        return x, y, theta

    def nearest_neighbor(self, tree, sample_x, sample_y, sample_theta):
        '''
        Find the nearest node in the given tree to the sample
        '''
        nearest_node = None
        min_dist = float('inf')
        for node in tree:
            dist = self.distance(node.x, node.y, sample_x, sample_y)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def steer(self, nearest, target_x, target_y, target_theta, max_extend_dist=0.5):
        '''
        Extend the nearest node towards the target point
        '''
        arc_rads = np.arange(-10, 10, 3)
        v_comms = np.arange(1, 10, 0.5)

        best_node = None
        best_dist = float('inf')

        for arc_rad in arc_rads:
            for v_comm in v_comms:
                # Compute left and right wheel velocities
                comm_velos = self.velocity_control(arc_rad, v_comm, nearest.theta)
                vl = comm_velos[0][0]
                vr = comm_velos[1][0]

                # Compute next position
                velo = (self.rad / 2) * (vl + vr)
                ang_velo = (self.rad / self.wheelbase) * (vr - vl)
                next_theta = nearest.theta + ang_velo * self.dt
                next_x = nearest.x + velo * np.cos(next_theta) * self.dt
                next_y = nearest.y + velo * np.sin(next_theta) * self.dt

                # Limit extension distance
                dist_to_target = self.distance(next_x, next_y, target_x, target_y)
                if dist_to_target > max_extend_dist:
                    continue

                if dist_to_target < best_dist:
                    best_dist = dist_to_target
                    best_node = RRTNode(next_x, next_y, next_theta, vl, vr, arc_rad, v_comm, nearest)

        return best_node

    def connect(self, tree, target_x, target_y, target_theta, max_attempts=50):
        '''
        Try to connect a tree to a target point
        '''
        for _ in range(max_attempts):
            nearest = self.nearest_neighbor(tree, target_x, target_y, target_theta)
            new_node = self.steer(nearest, target_x, target_y, target_theta)
            if new_node is not None:
                tree.append(new_node)
                if self.distance(new_node.x, new_node.y, target_x, target_y) <= self.goal_radius:
                    return new_node  # Successfully connected
        return None

    def plan(self):
        '''
        Main RRT-Connect Planning Loop
        '''
        start_node = RRTNode(self.start_x, self.start_y, 0, 0, 0, 0, 0, None)
        goal_node = RRTNode(self.goal_x, self.goal_y, 0, 0, 0, 0, 0, None)

        self.tree_start.append(start_node)
        self.tree_goal.append(goal_node)

        for _ in range(self.max_samples):
            # Random sample
            sample_x, sample_y, sample_theta = self.random_sample()

            # Extend the start tree
            nearest_start = self.nearest_neighbor(self.tree_start, sample_x, sample_y, sample_theta)
            new_start_node = self.steer(nearest_start, sample_x, sample_y, sample_theta)

            if new_start_node is not None:
                self.tree_start.append(new_start_node)

                # Try to connect the goal tree to the new start node
                connect_node = self.connect(self.tree_goal, new_start_node.x, new_start_node.y, new_start_node.theta)
                if connect_node is not None:
                    # Reconstruct the path
                    path_start = []
                    current = new_start_node
                    while current is not None:
                        path_start.append((current.x, current.y, current.theta, current.arc_rad, current.v_comm))
                        current = current.parent

                    path_goal = []
                    current = connect_node
                    while current is not None:
                        path_goal.append((current.x, current.y, current.theta, current.arc_rad, current.v_comm))
                        current = current.parent

                    return path_start[::-1] + path_goal  # Combine the two paths

            # Swap trees (start and goal alternate roles)
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start

        return None


if __name__ == '__main__':
    dt = 2                      # Time step
    rad = 0.325                 # Wheel Radius
    width = 1.64                # Axle Width
    wheelbase = 1.91            # Wheelbase of ZOE2
    goal_radius = 0.1           # Goal radius for termination
    max_samples = 1000          # Maximum number of iterations

    start_x, start_y = 0, 0    # Start position
    goal_x, goal_y = 4, 4      # Goal position

    planner = RRTConnectPlanner(start_x, start_y, goal_x, goal_y, [dt, rad, width, wheelbase, goal_radius, max_samples])
    plan = planner.plan()  # Plan with RRT-Connect
    print(plan)
