from calendar import c
from tkinter import RIGHT
from urllib import robotparser
from astar import AStarPlanner
#from bfs import BreadthFirstSearchPlanner
from matplotlib import pyplot as plt
from mouse import mouse_trajectory
from collision import intersects
import time
from typing import List, Tuple
from copy import copy
import numpy as np
import pygame
from math import sin, cos, tan, atan2, pi, sqrt
import cv2
WIDTH = 900
HEIGHT = 900

def euclidian(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

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
    width: int = 120
    height: int = 90
    angle = 0
    points = []
    dt = 0 


    def __init__(self, x, y, theta) -> None:
        self.x_coord = x
        self.y_coord = y
        self.angle = theta
        self.phi = 0
        self.vel_l = 0
        self.vel_r = 0
        self.max_turn = 45
        self.rad = self.width / tan(self.max_turn * pi/180)
        
        # self.chassis = self.robot_points()
    
    def get_robot_points(self):
        points = []
        
        points.append([self.width+0,0])
        points.append([self.width, self.height/2])
        points.append([0, self.height/2])
        points.append([0, -self.height/2])
        points.append([self.width, -self.height/2])
        

        # right side
        points.append([0, self.height/2])
        points.append([self.width, self.height/2])
        # left wheel
        points.append([0, -self.height/2])
        points.append([self.width, -self.height/2])

        # front wheels
        points.append(rot_points([-15,0], -self.phi) + np.array([self.width-20,self.height/2-10]))
        points.append(rot_points([15,0], -self.phi) + np.array([self.width-20,self.height/2-10]))
        points.append(rot_points([-15,0], -self.phi) + np.array([self.width-20,-self.height/2+10]))
        points.append(rot_points([15,0], -self.phi) + np.array([self.width-20,-self.height/2+10]))

        # normal line
        points.append(rot_points([0,141], -self.phi) + np.array([self.width,0]))
        points.append(np.array([self.width,0]))

        points.append([0, self.rad])
        points.append([0, -self.rad])


        points = rot_points(points, self.angle) + np.array([self.x_coord, self.y_coord])
        points = convert_to_display(points)
        
        return points

    def move(self, speed: float, phi: float = 0) -> None:
        self.phi = phi

        p = abs(self.phi * pi/180)
        w = self.height
        h = self.width
        vel_l = speed
        vel_r = vel_l * ((2*h - w*tan(p))/(2*h + w*tan(p)))

        if phi > 0:
            self.vel_r = vel_r
            self.vel_l = vel_l
        else:
            self.vel_l = vel_r
            self.vel_r = vel_l
    
        
    
    def turn(self, speed:float, gain: float, diff_heading: float) -> None:
        angle_thresh = self.max_turn
        # +ve is RIGHT
        # -ve is LEFT
        if diff_heading > 0:
            if diff_heading > 180:
                # turn right
                if gain*diff_heading > angle_thresh: self.move(speed, angle_thresh)
                else: self.move(speed, gain*diff_heading)
            else:
                # turn left
                if gain*diff_heading > angle_thresh: self.move(speed, -angle_thresh)
                else: self.move(speed, -gain*diff_heading)
        else:
            if diff_heading > -180:
                # turn right
                if gain*diff_heading < -angle_thresh: self.move(speed, angle_thresh)
                else: self.move(speed, -gain*diff_heading)
            else:
                # turn left
                if gain*diff_heading < -angle_thresh: self.move(speed, -angle_thresh)
                else: self.move(speed, gain*diff_heading)        
    
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

        self.robot.angle = self.robot.angle % 359
    
    def check_success(self, goal: Tuple[float, float]) -> bool:
        return np.allclose(self.robot.get_position(), goal, atol=5.5)
    

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
        self.obs += convert_to_display(np.array([[500,450]]))
        self.vehicle1 += convert_to_display(np.array([[700,75]]))
        self.vehicle2 += convert_to_display(np.array([[150,75]]))
        self.obstacles = [self.obs, self.vehicle1, self.vehicle2]
        
        
class Planner:    
    def __init__(self, robot: Robot, world: World) -> None:
        self.robot = robot
        self.world = world
        self.trajectory = [[self.robot.x_coord, self.robot.y_coord],
                            [100,200],
                            [350,400],
                            [700,300],
                            [700,100],
                            [700, 400]]

    def get_trajectory(self):
        return self.trajectory

    def get_heading(self, dx, dy) -> float:
        heading = atan2(dy,dx) * 180/pi
        # print("head: ", heading)
        if heading < 0:
            return 360 + heading
        else:
            return heading

    
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
        self.robot_path = []
    
    def display_robot(self):
        robot_points = self.robot.get_robot_points()
        pygame.draw.circle(self.screen, self.BLACK, (self.robot.x_coord, HEIGHT-self.robot.y_coord),5)
        # pygame.draw.circle(self.screen, self.BLACK, (self.robot.x_coord, HEIGHT-self.robot.y_coord),40,2)
        for i in range(5):
            if i == 4:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[0], 4)
            else:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[i+1], 4)
        
        # left wheel
        pygame.draw.circle(self.screen, self.BLUE, robot_points[5], 5)
        pygame.draw.circle(self.screen, self.BLUE, robot_points[6], 5)
        # right wheel
        pygame.draw.circle(self.screen, self.BLACK, robot_points[7], 5)
        pygame.draw.circle(self.screen, self.BLACK, robot_points[8], 5)

        # front wheel
        pygame.draw.line(self.screen, self.BLACK, robot_points[9], robot_points[10], 4)
        pygame.draw.line(self.screen, self.BLACK, robot_points[11], robot_points[12], 4)

        # normal line
        # pygame.draw.line(self.screen, self.BLUE, robot_points[11], robot_points[12], 2)

        self.robot_path.append([self.robot.x_coord, self.robot.y_coord])
        robot_path = convert_to_display(np.array(self.robot_path))

        for r in robot_path:
            pygame.draw.circle(self.screen, self.BLACK, r, 1)
        
        p1 = robot_points[-2]
        p2 = robot_points[-1]

        # pygame.draw.line(self.screen, self.GREEN, p1, (self.robot.x_coord, 900-self.robot.y_coord), 4)
        # pygame.draw.line(self.screen, self.GREEN, p2, (self.robot.x_coord, 900-self.robot.y_coord), 4)
        

        coor = 'Coordinates: ' + str(np.round(self.robot.get_position(),2))
        speeds = 'Speeds: ' + str(np.round(self.robot.get_speed(),2))
        angle = 'Steering: ' + str(np.round(self.robot.phi,2))
        self_angle = 'Heading: ' + str(np.round(self.robot.angle,2))
        text = self.font.render(coor, True, self.BLACK)
        self.screen.blit(text, (1, 30))
        text1 = self.font.render(speeds, True, self.BLACK)
        self.screen.blit(text1, (1, 5))
        text2 = self.font.render(angle, True, self.BLACK)
        self.screen.blit(text2, (1, 55))
        text3 = self.font.render(self_angle, True, self.BLACK)
        self.screen.blit(text3, (1, 80))


    def display_world(self, counter=0):
        # pygame.draw.circle(self.screen, self.BLACK, (self.world.width/2, self.world.height/2),5)
        for i in range(len(self.world.obs)-1):
            pygame.draw.line(self.screen, self.RED, self.world.obs[i], self.world.obs[i+1], 8)
        rect1 = (self.world.obs[2][0]-105, self.world.obs[2][1]-85, 570, 370)
        


        for i in range(len(self.world.vehicle1)-1):
            pygame.draw.line(self.screen, self.RED, self.world.vehicle1[i], self.world.vehicle1[i+1], 8)
        rect2 = (self.world.vehicle1[2][0]-55, self.world.vehicle1[2][1]-55, 310, 210)
        
        
        for i in range(len(self.world.vehicle2)-1):
            pygame.draw.line(self.screen, self.RED, self.world.vehicle2[i], self.world.vehicle2[i+1], 8)
        rect3 = (self.world.vehicle2[2][0]-75, self.world.vehicle2[2][1]-85, 350, 250)
        
        traj = convert_to_display(np.array(self.planner.get_trajectory()))

        if counter==-1:
            pygame.draw.rect(self.screen, self.BLACK, rect1, width=2, border_radius=40)
            pygame.draw.rect(self.screen, self.BLACK, rect2, width=2, border_radius=40)
            pygame.draw.rect(self.screen, self.BLACK, rect3, width=2, border_radius=40)
        # else:
        #     pygame.draw.circle(self.screen, self.GREEN, traj[-1], 20, 2)
        #     pygame.draw.circle(self.screen, self.BLACK, traj[-1], 5)

        pygame.draw.circle(self.screen, self.BLACK, traj[0], 5)
        # for i in range(counter-1):
        #     pygame.draw.line(self.screen, self.BLUE, traj[i], traj[i+1])
        
        # for i in range(len(traj)):
        #     pygame.draw.circle(self.screen, self.BLUE, traj[i], 3)
        
        


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
        running = True
        counter = 0
        last_counter = True
        lasttime = pygame.time.get_ticks() 
        success = False    
        is_colliding = False 
        goal = self.planner.get_trajectory()
        counter_loop = 0
        gain = 2.5
        point1 = False
        point2 = False

        while running:

            self.controller.step()

            if counter < len(goal):
                success = self.controller.check_success(goal[counter])
                goal_check = False
                # print("Counter: ", counter)
            else:
                success = True
                goal_check = True
                
                diff_heading = self.robot.angle
                if abs(diff_heading) > 0.5 and not point1:
                    # print("Final goal reached\r")
                    self.robot.turn(0.1, 1, self.robot.max_turn)
                    # point1 = False
                else:
                    self.robot.move(0,0)
                    point1 = True
                
                    if not is_colliding and not point2:
                        self.robot.move(0.1,0)
                    else:
                        point2 = True
                        if point1 and self.robot.y_coord > 100:
                            self.robot.turn(-0.1, 1, -self.robot.max_turn)
                        else:
                            if self.robot.angle > 0.5:
                                self.robot.turn(-0.1, 1, self.robot.max_turn)
                            else:
                                if is_colliding:
                                    self.robot.move(-1,0)
                                else:
                                    self.robot.move(0,0)

           
                
            if not goal_check:
                dy = goal[counter][1] - self.robot.y_coord
                dx = goal[counter][0] - self.robot.x_coord
                counter_loop += 1
                # print(counter_loop)


            heading = np.round(self.planner.get_heading(dx,dy),0)
            diff_heading = heading - self.robot.angle
            # print(diff_heading)

            if abs(diff_heading) > 0.5 and not goal_check:
                # print(counter)
                if counter < len(goal)-1:
                    points = self.robot.get_robot_points()
                    p1 = points[-2]
                    p2 = points[-1]
                    p3 = [goal[counter+1][0], 900-goal[counter+1][1]]
                    buffer = self.robot.rad+5
                    if (euclidian(p1, p3)<buffer or euclidian(p2, p3)<buffer) and counter!=1:
                        # print("Rad")
                        counter += 1
                        # print(counter)
                self.robot.turn(0.1, gain, diff_heading)
            
            elif not goal_check:
                # print("Move towards goal")
                speed = 0.1
                self.robot.move(speed, 0)
            
            if success and not goal_check:
                # do stuff for new goal
                # stop robot
                # if not goal_check:
                self.robot.move(0,0)
                # print("WAYPOINT")
                counter += 1
                gain = 2.5
                counter_loop = 0
                # else:
                #     self.robot.move(0,0)
                # print("Waypoint reached")
            else:
                if self.planner.is_collision():
                    is_colliding = True
                else:
                    is_colliding = False


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
    # path_planner = BreadthFirstSearchPlanner(ox, oy, image)

    trajectory = mouse_trajectory(world.obstacles, width, height)
    # gx = trajectory[-1][1]//3
    # gy = trajectory[-1][0]//3
    gy = 390//3
    gx = 760//3
    rx, ry = path_planner.planning(0,0, gx, gy)

    for i in range(len(rx)-1):
        image[rx[i]][ry[i]] = 127

    # cv2.imshow("map", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    traj = np.flip(np.vstack((ry*3,rx*3)).T, axis=0)
    trajectory = traj[1::1]
    print(trajectory[-1])
    trajectory = np.vstack((trajectory, traj[-1]))
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
