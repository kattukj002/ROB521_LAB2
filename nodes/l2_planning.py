#!/usr/bin/env python
#Standard Libraries
import numpy as np
from numpy.linalg import inv
import math
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import circle
from scipy.linalg import block_diag

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

def check_if_close_enough(x1, y1, x2, y2, threshold):

    close_enough = False

    if (abs(x1 - x2) < threshold) and (abs(y1 - y2) < threshold):
        close_enough = True

    return close_enough

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)

        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[1] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        x_upper = 45
        x_lower = -2
        y_upper = 11
        y_lower = -49

        x_sample = (np.random.rand(1,1) * (x_upper - x_lower)) + x_lower
        y_sample = (np.random.rand(1,1) * (y_upper - y_lower)) + y_lower

        return np.array([x_sample[0, 0], y_sample[0, 0], 0])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        threshold = 0.1
        is_duplicate = False

        #iterate over all the stored nodes
        for i in range(len(self.nodes)):
            curr_node = self.nodes[i]

            if (abs(point[0] - curr_node.point[0]) < threshold) and (abs(point[1] - curr_node.point[1]) < threshold):
                is_duplicate = True
                break

        return is_duplicate
    
    def closest_node(self, point):
        #Returns the index of the closest node

        min_dist = 10000000000
        closest_node_index = 0

        for i in range(len(self.nodes)):
            curr_node = self.nodes[i]

            #use Euclidean distance
            curr_dist = math.sqrt((curr_node.point[0] - point[0])**2 + (curr_node.point[1] - point[1])**2)

            if curr_dist < min_dist:
                min_dist = curr_dist
                closest_node_index = i

        return closest_node_index, min_dist
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]

        #initialize the robot trajectory to the initial point
        robot_traj_overall = np.zeros(shape=(3,1))
        robot_traj_overall[0] = node_i.point[0][0]
        robot_traj_overall[1] = node_i.point[1][0]
        robot_traj_overall[2] = node_i.point[2][0]
        
        current_point = Node(node_i.point, node_i.parent_id, node_i.cost)

        goal_reached = False

        curr_iteration_num = 0

        while (not goal_reached):

            if curr_iteration_num % 10000 == 0:
                print("This is current iteration: " + str(curr_iteration_num))
                print("This is current point: ")
                print(current_point.point)
                print("This is goal: ")
                print(point_s)
                print("\n\n")

            #check if goal has been reached
            if check_if_close_enough(current_point.point[0], current_point.point[1], point_s[0], point_s[1], 0.5):
                print("You are close enough")
                print("This is current point: ")
                print(current_point.point)
                print("This is goal: ")
                print(point_s)
                goal_reached = True
                break

            #get v, omega needed to get to point_s
            vel, rot_vel  = self.robot_controller(current_point, point_s)

            #simulate next 10s of v, omega
            robot_traj = self.trajectory_rollout(vel, rot_vel, current_point)

            #iterate over the 10 timesteps in the trajectory
            goal_within_trajectory = False
            for i in range(robot_traj.shape[1]):

                if check_if_close_enough(robot_traj[0, i], robot_traj[1, i], point_s[0], point_s[1], 0.1*self.vel_max):
                    goal_within_trajectory = True
                    robot_traj_overall = np.hstack((robot_traj_overall, robot_traj[:, :i]))

                    current_point_pose = np.array([[robot_traj[0, i]], [robot_traj[1, i]], [robot_traj[2, i]]])

                    current_point = Node(current_point_pose, -1, 0)
                    break

            if not(goal_within_trajectory):
                robot_traj_overall = np.hstack((robot_traj_overall, robot_traj))
                current_point_pose = np.array([[robot_traj[0, -1]], [robot_traj[1, -1]], [robot_traj[2, -1]]])
                current_point = Node(current_point_pose, -1, 0)

            curr_iteration_num = curr_iteration_num + 1

        return robot_traj_overall
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        controller_number = 1

        if controller_number == 1:
            curr_robot_angle = node_i.point[2]
            curr_x = node_i.point[0]
            curr_y = node_i.point[1]
            desired_x = point_s[0]
            desired_y = point_s[1]

            #positive constant gain terms
            k1 = 0.1
            k2 = 0.45
            k3 = 0.1   #makes the values change more

            #desired end point values
            v_r = 0.55
            omega_r = 0
            theta_r = 0

            v = v_r*math.cos(theta_r - curr_robot_angle) + k1*(desired_x - curr_x)
            omega = omega_r - k2*v_r*np.sinc(theta_r-curr_robot_angle)*(desired_y - curr_y) + k3*(theta_r-curr_robot_angle)

        elif controller_number == 2:
            #robot stuff
            curr_robot_angle = node_i.point[2]
            curr_robot_x = node_i.point[0]
            curr_robot_y = node_i.point[1]

            # print("This is curr_robot_angle: ")
            # print(curr_robot_angle)
            # print("This is curr_robot_x: ")
            # print(curr_robot_x)
            # print("This is curr_robot_y: ")
            # print(curr_robot_y)

            goal_robot_x = point_s[0]
            goal_robot_y = point_s[1]

            # print("This is goal_robot_x")
            # print(goal_robot_x)
            # print("This is goal_robot_y")
            # print(goal_robot_y)

            R = np.array([[np.cos(curr_robot_angle)[0], -np.sin(curr_robot_angle)[0]], [np.sin(curr_robot_angle)[0], np.cos(curr_robot_angle)[0]]])

            curr_robot_position = np.array([[curr_robot_x][0], [curr_robot_y][0]])
            goal_robot_position = np.array([[goal_robot_x], [goal_robot_y]])

            # print("This is curr_robot_position")
            # print(curr_robot_position)
            # print("This is shape of curr_robot_position")
            # print(curr_robot_position.shape)
            # print("This is goal_robot_position")
            # print(goal_robot_position)
            # print("This is shape of goal_robot_position")
            # print(goal_robot_position.shape)


            #algorithm stuff
            d = 0.1
            k1 = 0.05
            k2 = 0.05
            upper_delta = np.diag([1, -d])
            lower_delta = np.array([[d], [0]])
            K = np.diag([k1, k2])

            # print("This is R: ")
            # print(R)
            # print("This is shape of R: ")
            # print(R.shape)
            # print("This is position difference: ")
            # print(goal_robot_position - curr_robot_position)
            # print("This is shape of position difference: ")
            # print((goal_robot_position - curr_robot_position).shape)


            position_error = np.matmul(R, goal_robot_position - curr_robot_position)

            RHS = -np.matmul(K, np.tanh(position_error - lower_delta))

            velocities = np.matmul(inv(upper_delta), RHS)

            # print("This is velocities: ")
            # print(velocities)
            # print("This is shape of velocities")
            # print(velocities.shape)

            v = velocities[0]
            omega = velocities[1]

            # print("This is v: ")
            # print(v)
            # print("This is omega")
            # print(omega)

        if v > self.vel_max:
            v = self.vel_max

        if omega > self.rot_vel_max:
            omega = self.rot_vel_max

        return v, omega

        #START OLD ROBOT CONTROLLER CODE
        #constant linear velocity, 10% of max velocity
        # v = self.vel_max * 0.1

        # current_heading = round(node_i.point[2], 2)
        # desired_heading = round(math.atan2(point_s[1] - node_i.point[1], point_s[0] - node_i.point[0]), 2)

        # omega_scaling_factor = 0

        # #factor for scaling angular velocity
        # if abs(current_heading - desired_heading) > math.pi:
        #     omega_scaling_factor = (2*math.pi - abs(current_heading - desired_heading)) / math.pi
        # else:
        #     omega_scaling_factor = abs(current_heading - desired_heading) / math.pi

        # rot_vel = self.rot_vel_max * omega_scaling_factor

        # angle_difference = desired_heading - current_heading

        # if (angle_difference > math.pi and angle_difference < 2*pi) or (angle_difference >= -math.pi and angle_difference < 0):
        #     rot_vel *= -1
        
        # return v, rot_vel
        #END OLD ROBOT CONTROLLER CODE
    
    def trajectory_rollout(self, vel, rot_vel, current_point):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        trajectory_rollout_values = np.zeros((3, self.num_substeps))

        for timestep in range(self.num_substeps):
            if rot_vel == 0:
                x = vel*math.cos(current_point.point[2])*timestep + current_point.point[0]
                y = vel*math.sin(current_point.point[2])*timestep + current_point.point[1]
                theta = current_point.point[2]
            else:
                x = (vel/rot_vel) * math.sin(rot_vel*timestep) + current_point.point[0]
                y = (vel/rot_vel) * (1 - math.cos(rot_vel*timestep)) + current_point.point[1]
                theta =  rot_vel*timestep + current_point.point[2]

            trajectory_rollout_values[0, timestep] = x
            trajectory_rollout_values[1, timestep] = y
            trajectory_rollout_values[2, timestep] = theta

        return trajectory_rollout_values
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        N = point.shape[1]
        pixel_point = np.empty((2, N))
        for i in range(N):
            pixelx = ((point[0,i] - (-21))/80) * 1600
            pixely = ((point[1,i] + 30.75)/80) * 1600
            pixel_point[0,i] = pixelx
            pixel_point[1,i] = pixely

        print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        return (pixel_point)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        rr_values = []
        cc_values = []
        pixel_point = self.point_to_cell(points)

        for i in range(pixel_point.shape[1]):
            rr, cc = circle(pixel_point[0,i], pixel_point[1,i], self.robot_radius)
            rr_values.append(rr)
            cc_values.append(cc)

        return (rr_values,cc_values)
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(500): #Most likely need more iterations than this to complete the map!
            
            is_point_same_as_existing_node = True

            point = np.array([0, 0, 0])

            #keep sampling until you find a point that's not the same as an existing node
            while (is_point_same_as_existing_node):
                #Sample map space
                point = self.sample_map_space()

                if not(self.check_if_duplicate(point)):
                    is_point_same_as_existing_node = False
            
            #Get the closest point
            closest_node_id, closest_node_dist = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            #Check for collisions
            
            #construct the 2xN matrix of x,y points
            in_colission = False
            robot_trajectory_points = trajectory_o[:2, :]
            robot_pixel_coords_x, robot_pixel_coords_y = self.points_to_robot_circle(robot_trajectory_points)

            #iterate over the timesteps in the trajectory
            for j in range(len(robot_pixel_coords_x)):

                #get x and y pixel coordinates at the current trajectory point
                curr_x_coords = robot_pixel_coords_x[j]
                curr_y_coords = robot_pixel_coords_y[j]

                #iterate over all the x,y pairs in curr_x_coords
                for k in range(len(curr_x_coords)):
                    curr_x = curr_x_coords[k]
                    curr_y = curr_y_coords[k]

                    #if there is a colission
                    if self.occupancy_map[curr_x, curr_y] == 0:
                        in_colission = True
                        break

                if in_colission == True:
                    break
            
            if not(in_colission):
                new_node = Node(np.array([[point[0]], [point[1]], [point[2]]]), closest_node_id, self.nodes[closest_node_id].cost + closest_node_dist)
            
            #Check if goal has been reached
            if check_if_close_enough(point[0], point[1], self.goal_point[0], self.goal_point[1], 0.5):
                break

        print("Finished the RRT planning function")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id, closest_node_dist = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    #map_filename = "simple_map.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    # nodes = path_planner.rrt_star_planning()
    nodes = path_planner.rrt_planning()

    #add nodes to window to visualize RRT planner
    for node in nodes:
        self.window.add_point(node)

    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()

