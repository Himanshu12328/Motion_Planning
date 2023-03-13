from astar import AStarPlanner
#from bfs import BreadthFirstSearchPlanner
from matplotlib import pyplot as plt
from mouse import mouse_trajectory
from collision import intersects
import time
from typing import List, Tuple, Union
from copy import copy
import numpy as np
import pygame
import random
import math
from math import sin, cos, atan, atan2, pi, sqrt
import cv2
WIDTH = 900
HEIGHT = 900

def max_speed(speed):
    if speed > 1:
        return 1
    else:
        return speed
    
def max_turn(speed):
    thresh = 0.1
    if abs(speed) > thresh:
        return thresh
    else:
        return abs(speed)


def rot_points(mat, degrees: float):
    rot = []
    degrees = degrees * pi/180
    rot_mat = np.array([[cos(degrees), sin(degrees)],[-sin(degrees), cos(degrees)]])

    return mat @ rot_mat

def convert_to_display(mat):
    mat[:,1] = HEIGHT - mat[:,1]
    return mat

class Robot:
    x_coord: int
    y_coord: int
    vel_r: int
    vel_l: int
    width: int = 20*2
    height: int = 30*2
    angle = 0
    points = []
    dt = 0 


    def __init__(self, x, y, theta) -> None:
        self.x_coord = x
        self.y_coord = y
        self.angle = theta
        self.vel_l = 0
        self.vel_r = 0
        
        # self.chassis = self.robot_points()
    
    def get_robot_points(self):
        points = []
        
        points.append([self.width+10,0])
        points.append([self.width/2, self.height/2])
        points.append([-self.width/2, self.height/2])
        points.append([-self.width/2, -self.height/2])
        points.append([self.width/2, -self.height/2])
        

        # right wheel
        points.append([0, self.height/2])
        # left wheel
        points.append([0, -self.height/2])

        
        points = rot_points(points, self.angle) + np.array([self.x_coord, self.y_coord])
        points = convert_to_display(points)
        
        return points

    def move(self, vel_l, vel_r) -> None:
        self.vel_r = vel_r
        self.vel_l = vel_l
    
    def turn(self, diff_heading, turn_speed, min_speed=0) -> None:
        if diff_heading >= 0:
            if diff_heading <=180:
                self.move(turn_speed + min_speed, -turn_speed + min_speed)
            else:
                self.move(-turn_speed + min_speed, turn_speed + min_speed)
        else:
            if diff_heading <= -180:
                self.move(turn_speed + min_speed, -turn_speed + min_speed)
            else:
                self.move(-turn_speed + min_speed, turn_speed + min_speed)
    
    def get_position(self) -> Tuple[float, float]:
        return self.x_coord, self.y_coord
    
    def get_speed(self) -> Tuple[float, float]:
        return self.vel_l, self.vel_r

    def set_position(self, pos: Tuple[float, float]) -> None:
        self.x_coord = pos[0]
        self.y_coord = pos[1]

        

class Controller:

    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    def step(self):
        # theta = pi - self.robot.angle * pi/180
        theta = self.robot.angle * pi/180
        self.robot.x_coord += ((self.robot.vel_l+self.robot.vel_r)/2)*cos(theta) * self.robot.dt
        self.robot.y_coord += ((self.robot.vel_l+self.robot.vel_r)/2)*sin(theta) * self.robot.dt
        self.robot.angle += atan2((self.robot.vel_r - self.robot.vel_l),self.robot.height)*180/pi * self.robot.dt

        self.robot.angle = self.robot.angle % 360
    
    def check_success(self, goal: Tuple[float, float]) -> bool:
        return np.allclose(self.robot.get_position(), goal, atol=3.5)
    

class World:    
    obs = np.array([[200,100],
                    [-200,100],
                    [-200,-100],
                    [200,-100],
                    [200,100]])
    vehicle1 = np.array([[100,50],
                    [-100,50],
                    [-100,-50],
                    [100,-50],
                    [100,50]])
    vehicle2 = np.array([[100,50],
                    [-100,50],
                    [-100,-50],
                    [100,-50],
                    [100,50]])

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.obs += convert_to_display(np.array([[450,400]]))
        self.vehicle1 += convert_to_display(np.array([[600,100]]))
        self.vehicle2 += convert_to_display(np.array([[250,100]]))
        self.obstacles = [self.obs, self.vehicle1, self.vehicle2]
        
        
class Planner:    
    def __init__(self, robot: Robot, world: World) -> None:
        self.trajectory = [[-100,-100]]
        self.robot = robot
        self.world = world

    def get_trajectory(self):
        return self.trajectory

    def get_heading(self, dx, dy) -> float:
        heading = atan2(dy,dx) * 180/pi
        # print("head: ", heading)
        if heading < 0:
            return 360 + heading
        else:
            return heading

    # if robot waypoint is in collision skip that waypoint
    def is_collision(self) -> bool:
        # get robot segments
        points = self.robot.get_robot_points()

        seg1 = (points[0], points[1])
        seg2 = (points[1], points[2])
        seg3 = (points[2], points[3])
        seg4 = (points[3], points[4])
        seg5 = (points[4], points[0])
        
        # check if any collide
        for obs in self.world.obstacles:
            for i in range(len(obs)):
                if i==len(obs)-1:
                    o = (obs[i], obs[0])
                else:
                    o = (obs[i], obs[i+1])
                if intersects(o, seg1) or intersects(o, seg2) or intersects(o, seg3) or \
                        intersects(o, seg4) or intersects(o, seg5):
                    return True
    
    def get_map(self, grid):
        h = int(HEIGHT/3)
        w = int(WIDTH/3)
        grid_map = np.zeros((h, w))
        counter = 1
        for i in range(h):
            for j in range(w):
                # print("Iter: ", counter,"sum: ", np.sum(grid[:,i,j]))
                counter += 1
                if np.sum(grid[:,i,j]) > 255:
                    grid_map[j][i] = 1
        return grid_map



class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)
    GREEN: Tuple[int, int, int] = (0, 255, 0)

    def __init__(self, robot: Robot, world: World, planner: Planner) -> None:
        pygame.init()
        pygame.font.init()
        self.robot = robot
        self.world = world
        self.planner = planner
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Farmland')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)
    
    def display_robot(self):
        robot_points = self.robot.get_robot_points()
        pygame.draw.circle(self.screen, self.BLACK, (self.robot.x_coord, HEIGHT-self.robot.y_coord),5)
        # pygame.draw.circle(self.screen, self.BLACK, (self.robot.x_coord, HEIGHT-self.robot.y_coord),40,2)
        for i in range(5):
            if i == 4:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[0], 4)
            else:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[i+1], 4)
        
        # right wheel
        pygame.draw.circle(self.screen, self.BLUE, robot_points[5], 5)
        # left wheel
        pygame.draw.circle(self.screen, self.BLACK, robot_points[6], 5)

        coor = 'Coordinates: ' + str(np.round(self.robot.get_position(),2))
        speeds = 'Speeds: ' + str(np.round(self.robot.get_speed(),2))
        text = self.font.render(coor, True, self.BLACK)
        self.screen.blit(text, (1, 30))
        text1 = self.font.render(speeds, True, self.BLACK)
        self.screen.blit(text1, (1, 5))


    def display_world(self, counter=0):
        # pygame.draw.circle(self.screen, self.BLACK, (self.world.width/2, self.world.height/2),5)
        for i in range(len(self.world.obs)-1):
            pygame.draw.line(self.screen, self.RED, self.world.obs[i], self.world.obs[i+1], 8)
        rect1 = (self.world.obs[2][0]-40, self.world.obs[2][1]-40, 480, 280)
        
        for i in range(len(self.world.vehicle1)-1):
            pygame.draw.line(self.screen, self.RED, self.world.vehicle1[i], self.world.vehicle1[i+1], 8)
        rect2 = (self.world.vehicle1[2][0]-40, self.world.vehicle1[2][1]-40, 280, 180)
        
        for i in range(len(self.world.vehicle2)-1):
            pygame.draw.line(self.screen, self.RED, self.world.vehicle2[i], self.world.vehicle2[i+1], 8)
        rect3 = (self.world.vehicle2[2][0]-40, self.world.vehicle2[2][1]-40, 280, 180)
        
        traj = convert_to_display(np.array(self.planner.get_trajectory()))

        if counter==-1:
            pygame.draw.rect(self.screen, self.BLACK, rect1, width=2, border_radius=40)
            pygame.draw.rect(self.screen, self.BLACK, rect2, width=2, border_radius=40)
            pygame.draw.rect(self.screen, self.BLACK, rect3, width=2, border_radius=40)
        else:
            pygame.draw.circle(self.screen, self.GREEN, traj[-1], 20, 2)
            pygame.draw.circle(self.screen, self.BLACK, traj[-1], 5)

        
        for i in range(counter-1):
            pygame.draw.line(self.screen, self.BLUE, traj[i], traj[i+1])

    def get_world_map(self):
        self.screen.fill(self.WHITE)
        self.display_world(-1)
        pygame.display.flip()
        map = np.array(pygame.surfarray.array2d(self.screen))
        map = np.swapaxes(map, 0, 1)
        return map.astype(np.uint8)


    def update_display(self, is_colliding, counter) -> bool:

        self.screen.fill(self.WHITE)

        self.display_world(counter)

        self.display_robot()

        for event in pygame.event.get():
            # Keypress
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # if escape is pressed, quit the program
                    return False
        
        if is_colliding:
            collide = self.font.render("Collision", True, self.BLACK)
        else:
            collide = self.font.render("No Collision", True, self.BLACK)
        self.screen.blit(collide, (500, 5))
        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, robot: Robot, controller: Controller, world: World, planner: Planner, vis: Visualizer) -> None:
        self.robot = robot
        self.controller = controller
        self.world = world
        self.planner = planner
        self.vis = vis      
        

    def run(self):
        # self.planner.trajectory = convert_to_display(np.array(trajectory))
        running = True
        counter = 0
        last_counter = -1
        lasttime = pygame.time.get_ticks() 
        success = False    
        is_colliding = False 
        goal = self.planner.get_trajectory()

        # main while loop
        while running:

            self.controller.step()


            if counter < len(goal):
                success = self.controller.check_success(goal[counter])
                goal_check = False
            else:
                success = True
                goal_check = True
                print("Final goal reached\r")
                diff_heading = self.robot.angle
                if abs(diff_heading) > 0.05:
                    gain = 0.01
                    turn_speed = max_turn(gain * diff_heading)
                    self.robot.turn(diff_heading, turn_speed)
                else:
                    time.sleep(5)
                    break

            if success:
                # do stuff for new goal
                # stop robot
                min_speed = 0.1
                if not goal_check:
                    self.robot.move(min_speed,min_speed)
                    counter += 1  
                              
                # print("Waypoint reached")
            else:
                if self.planner.is_collision():
                    is_colliding = True
                else:
                    is_colliding = False
                
                if counter != last_counter:
                    # angle to the next waypoint
                    dy = goal[counter][1] - self.robot.y_coord
                    dx = goal[counter][0] - self.robot.x_coord
                    last_counter = counter
                
                heading = np.round(self.planner.get_heading(dx,dy),0)
                diff_heading = self.robot.angle - heading
                
                if abs(diff_heading) > 0.05:
                    gain = 0.01
                    turn_speed = max_turn(gain * diff_heading)
                    self.robot.turn(diff_heading, turn_speed)
                else:
                    # self.robot.angle = heading
                    gain = 0.01
                    speed = max_speed(gain*((self.robot.x_coord-goal[counter][0])**2 + (self.robot.y_coord-goal[counter][1])**2)**0.5)
                    self.robot.move(speed, speed)
                    

            # dt            
            self.robot.dt = (pygame.time.get_ticks() - lasttime)
            lasttime = pygame.time.get_ticks()

            running = self.vis.update_display(is_colliding, counter)
            
            time.sleep(0.001)
        

def main():
    height = HEIGHT
    width = WIDTH


    robot = Robot(0,900,0)
    controller = Controller(robot)
    world = World(width, height)
    planner = Planner(robot, world)
    vis = Visualizer(robot, world, planner)

    world_map = vis.get_world_map()
    image = cv2.resize(255 - world_map, (300,300), interpolation=cv2.INTER_AREA)
    

    obs_idx = np.argwhere(image == 255)
    ox = np.array(obs_idx[:,0])
    oy = np.array(obs_idx[:,0])

    path_planner = AStarPlanner(ox, oy, image)
    # planner = BreadthFirstSearchPlanner(ox, oy, image)

    trajectory = mouse_trajectory(world.obstacles, width, height)
    gx = trajectory[-1][1]//3
    gy = trajectory[-1][0]//3
    rx, ry = path_planner.planning(0,0, gx, gy)

    for i in range(len(rx)-1):
        image[rx[i]][ry[i]] = 127

    # cv2.imshow("map", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    traj = np.flip(np.vstack((ry*3,rx*3)).T, axis=0)
    trajectory = traj[1::10]

    planner.trajectory = convert_to_display(np.array(trajectory))

    runner = Runner(robot, controller, world, planner, vis)

    try:
        runner.run()
        pass
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass
    finally:
        vis.cleanup()


if __name__ == '__main__':
    main()