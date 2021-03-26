# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovariance, PoseWithCovarianceStamped, PointStamped, Point32
from sensor_msgs.msg import PointCloud
from vi_navigation_interface.msg import PlannedPath, PlannedActions
from rosgraph_msgs.msg import Clock

import tf2_ros

import os
from functools import partial
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
import time
from .transformations import euler_from_quaternion, quaternion_from_euler

import pickle
from datetime import datetime

pi = 3.1415927410125732

@torch.jit.script
def action_transforme(a_t):
    # scaling parameters for the action
    a_support = torch.tensor([2*0.22,2*2.84], dtype=torch.float) # [m/s,rad/s] turtlebot3 burger
    a_offset = -a_support/2

    return a_support*a_t + a_offset

@torch.jit.script
def f(z_t, a_t, deltaT):
    # Simple Kinematic Motion model for uni-cycle robot with linear/angular velocity (Twist) input
    # The model is dicretized by the Euler method
    a = action_transforme(a_t)

    z_tp1 = torch.empty(3)

    z_tp1[0] = z_t[0] + torch.cos(z_t[2])*a[0]*deltaT[0]
    z_tp1[1] = z_t[1] + torch.sin(z_t[2])*a[0]*deltaT[0]
    z_tp1[2] = z_t[2] + a[1]*deltaT[0]

    return z_tp1

def F(z_t, a_t, deltaT):
    z_t_p1 = f(z_t, a_t, deltaT)

    M = deltaT*torch.tensor([0.05,0.05,0.10],dtype=torch.float)

    return dist.Uniform(z_t_p1-M, z_t_p1+M)

@torch.jit.script
def constraint(z_t,m):
    min_dist = m[torch.tensor([2])]
    dist = torch.dist(m[torch.tensor([0, 1])], z_t[torch.tensor([0, 1])],p=2)
    if dist<= min_dist:
        return torch.tensor([1.],dtype=torch.float)
    else:
        dist = dist - min_dist
        return torch.exp(-dist*25) # the higher constant the closer we allow the robots


def decisionModel(self, z_meas_dist, a_current, deltaT, z_goal, knownPlannedPaths, planning_time, time_left_from_predicted_pose, T1, T2):

    with pyro.util.ignore_jit_warnings():
        a = torch.tensor([0.,0.],dtype=torch.float)
        b = torch.tensor([1.,1.],dtype=torch.float)
        constraintObs = torch.tensor([0.],dtype=torch.float)
        costObs = torch.tensor([1.],dtype=torch.float)

    z_meas = pyro.sample("z_meas_{}".format(T1), z_meas_dist)
    z_predicted = pyro.sample("z_{}_plus".format(T1), F(z_meas, a_current, planning_time))

    if len(knownPlannedPaths) != 0: # check if we have received planned trajectories from other robots
        z = {}
        for ID in knownPlannedPaths:
            with pyro.util.ignore_jit_warnings():
                z[ID] = torch.zeros(T2-T1,3)
            z[ID] = motionModel(knownPlannedPaths[ID]["z_meas_dist"], 
                                knownPlannedPaths[ID]["a_current"], 
                                knownPlannedPaths[ID]["a_alpha"], 
                                knownPlannedPaths[ID]["a_beta"], 
                                deltaT, 
                                knownPlannedPaths[ID]["planning_time"], 
                                knownPlannedPaths[ID]["time_left_from_predicted_pose"],
                                T1, T2)

    for tau in pyro.markov(range(T1, T2)):
        a_prev = pyro.sample("a_{}".format(tau), dist.Uniform(a,b).to_event(1)) # the actions are scaled in the model function!
        if tau == T1:
            z_new = pyro.sample("z_{}".format(tau+1),F(z_predicted, a_prev, time_left_from_predicted_pose))
        else:
            z_new = pyro.sample("z_{}".format(tau+1),F(z_prev, a_prev, deltaT))
        
        if len(knownPlannedPaths) != 0:
            for ID in knownPlannedPaths:
                m_ID_t = z[ID][tau-T1,:]
                m_ID_t[torch.tensor([2])] = self.radius + knownPlannedPaths[ID]["radius"] # NOTICE HOW THE ANGLE is written over...
                pyro.sample("c_{}_{}".format(ID,tau+1), dist.Bernoulli(constraint(z_new,m_ID_t)), obs=constraintObs) # constraint!

        if tau <= T2-1:
            C = torch.dist(z_new[[0,1]],z_goal.index_select(0, torch.tensor([0, 1])),p=2)
            pyro.sample("o_{}".format(tau+1), dist.Bernoulli(torch.exp(-3.0*C)), obs=costObs) # optimality function
        
        z_prev = z_new


def decisionGuide(self, z_meas_dist, a_current, deltaT, z_goal, knownPlannedPaths, planning_time, time_left_from_predicted_pose, T1, T2):

    with pyro.util.ignore_jit_warnings():
        alpha = torch.tensor([100., 1000.],dtype=torch.float) # small preference for going forward initially 
        beta = torch.tensor([10., 1000.],dtype=torch.float)
        z = torch.zeros(T2-T1,3)

    z_meas = pyro.sample("z_meas_{}".format(T1), z_meas_dist)
    z_predicted = pyro.sample("z_{}_plus".format(T1), F(z_meas, a_current, planning_time))

    for tau in pyro.markov(range(T1, T2)):
        a_alpha_t = pyro.param("a_alpha_{}".format(tau), alpha,constraint=constraints.positive) # alpha,beta = 1 gives uniform!
        a_beta_t = pyro.param("a_beta_{}".format(tau), beta,constraint=constraints.positive) # alpha,beta = 1 gives uniform!
        a_prev = pyro.sample("a_{}".format(tau), dist.Beta(a_alpha_t, a_beta_t).to_event(1))
        if tau == T1:
            z_new = pyro.sample("z_{}".format(tau+1),F(z_predicted, a_prev, time_left_from_predicted_pose))
        else:
            z_new = pyro.sample("z_{}".format(tau+1),F(z_prev, a_prev, deltaT))
        
        z[tau-T1,:] = z_new

        z_prev = z_new

    return z

#@torch.jit.script
def motionModel(z_meas_dist, a_current, a_alpha, a_beta, deltaT, planning_time, time_left_from_predicted_pose, T1, T2):
    z = torch.zeros(T2-T1,3)

    z_meas = z_meas_dist.sample()
    z_predicted = F(z_meas, a_current, planning_time).sample()

    for tau in range(T1, T2):
        a_prev = dist.Beta(a_alpha[tau-T1,:], a_beta[tau-T1,:]).sample()
        if tau == T1:
            z_new = F(z_predicted, a_prev, time_left_from_predicted_pose).sample()
        else:
            z_new = F(z_prev, a_prev, deltaT).sample()
        
        z[tau-T1,:] = z_new

        z_prev = z_new

    return z


@torch.jit.script
def getNormalMarginal(mean, cov, indices):
    # mean: mean of the normal distribution as a vector
    # cov: cov of the normal distribution either as a NxN matrix or a 1xN or Nx1 vector in row major representation
    mean_marginal = mean[indices]

    size = torch.tensor(cov.shape,dtype=torch.float32)

    if size[0] != size[1]:
        if size[0]==1 or size[1]==1:
            dim = int(torch.sqrt(torch.max(size)))
            cov = torch.reshape(cov, (dim, dim)) # 6x6 matrix
        #else:
            # error...

    cov_marginal = torch.index_select(cov, 0, indices)
    cov_marginal = torch.index_select(cov_marginal, 1, indices) # covariance matrix for marginal distribution of [x, y, yaw]

    return mean_marginal, cov_marginal

@torch.jit.script
def expandCovarianceMatrix(M, newShape, newIndices):
    M_new = torch.diag(0.001*torch.ones(newShape[0]))
    for i in range(M.shape[0]):
        idx1 = newIndices[i]
        for j in range(M.shape[1]):
            idx2 = newIndices[j]
            M_new[idx1,idx2] = M[i,j]

    return M_new

def alignTimer(clock,timer):
    # This function is a simple approximate alignment of a timer, it can be made smarter by compensating ns1 
    # for the time it takes to reset the timer. This would have to be done by running this
    # algorithm a couple of times and take the avarage of the error and subtract this from ns1

    dt = timer.timer_period_ns
    time1 = clock.now()
    ns = time1.nanoseconds
    N,mod = divmod(ns, dt)

    ns1 = (N+1)*dt # time for which the timer should be reset

    if ns == 0: # when sim_time_true, clock is always zero in the constructor... https://answers.ros.org/question/356733/ros2-why-rclcpp-node-now-function-returns-0-in-node-construtor/
        t0 = Time(nanoseconds=0)
    else:
        while True: # wait until one dt has passed...
            time2 = clock.now()
            ns2  = time2.nanoseconds
            if ns2 >= ns1: # reset timer
                timer.reset()
                break
        t0 = Time(nanoseconds=ns1) # return the time for when the timer was supposed to be reset
    return t0

class TrajectoryPlanner(Node):

    def __init__(self):
        super().__init__('velocity_publisher')

        self.get_logger().info('Initializing Co-operative shuttlebus controller')
        # self.group = ReentrantCallbackGroup() # to allow callbacks to be executed in parallel
        self.group = MutuallyExclusiveCallbackGroup()

        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.get_logger().info("use_sim_time: " + str(self.use_sim_time))

        self.ROBOTNAMESPACE = os.environ['ROBOTNAMESPACE']

        self.rviz = True
        self.Log_DATA = True

        # size of robots as a radius of minimum spanning disk 
        # https://emanual.robotis.com/docs/en/platform/turtlebot3/specifications/
        # the wheels on the robot is placed in the front. The robot is 138mm long and 178mm wide, thus asssuming the wheel to be placed in the front an conservative estimate is
        # from https://www.robotis.us/turtlebot-3-burger-us/ a more appropriate value would be 105 mm
        self.radius = 0.138

        self.z_meas = torch.tensor([0,0,0],dtype=torch.float)
        self.deltaT = torch.tensor([1.0],dtype=torch.float) # in seconds
        self.Tau_span = 4 # prediction horizon
        self.svi_epochs = 10
        if self.use_sim_time:
            self.covariance_scale = torch.tensor([1.0],dtype=torch.float)
        else:
            self.covariance_scale = torch.tensor([0.1],dtype=torch.float)# the covariances from AMCL seems very conservative - probably due to a need for population error in the particle filter. thus we scale it to be less conservative

        self.r_plan = 0.2

        if self.ROBOTNAMESPACE == "turtlebot1":
            self.z_plan = torch.tensor([[2.0000, 1.0000, 0.0000, self.r_plan],
                                        [0.0000, 1.0000, 0.0000, self.r_plan]],dtype=torch.float)
        elif self.ROBOTNAMESPACE == "turtlebot2":
            self.z_plan = torch.tensor([[1.5000, 1.8660, 0.0000, self.r_plan],
                                        [0.5000, 0.1340, 0.0000, self.r_plan]],dtype=torch.float)
        elif self.ROBOTNAMESPACE == "turtlebot3":
            self.z_plan = torch.tensor([[0.5000, 1.8660, 0.0000, self.r_plan],
                                        [1.5000, 0.1340, 0.0000, self.r_plan]],dtype=torch.float)
        else:
            self.z_plan = torch.tensor([[2.99,2.57,0,self.r_plan],
                                        [2.01,1.93,0,self.r_plan],
                                        [1.65,1.12,0,self.r_plan],
                                        [1.25,0.65,0,self.r_plan],
                                        [0.039,-0.18,0,self.r_plan],
                                        [1.25,0.65,0,self.r_plan],
                                        [1.65,1.12,0,self.r_plan],
                                        [2.01,1.93,0,self.r_plan]],dtype=torch.float)


        self.i_plan = 0

        self.meas_L = torch.diag(torch.tensor([1.,1.,1.],dtype=torch.float)) # cholesky decomposition of measurement model covariance - this is just a temporary assignment it is reassigned in the meas callback

        # setup the inference algorithm
        optim_args = {"lr":0.05}
        optimizer = pyro.optim.Adam(optim_args)
        self.svi = pyro.infer.SVI(model=decisionModel,
                            guide=decisionGuide,
                            optim=optimizer,
                            loss=pyro.infer.Trace_ELBO(num_particles=1))

        self.get_logger().info('Finished SVI setup')

        self.est_planning_epoch_time = Duration(nanoseconds=(self.deltaT[0]/3.)*10**9) #variable to save Cumulative moving average of execution times
        self.N_samples_epoch_time = 1

        self.a_current = torch.tensor([0.0,0.0],dtype=torch.float)
        self.a_next = torch.tensor([0.0,0.0],dtype=torch.float) # variable to save the current estimate of next action

        if not self.use_sim_time: # to get current pose from tf tree
            self.tfbuffer = tf2_ros.Buffer()
            self.tflistener = tf2_ros.TransformListener(self.tfbuffer, self)

        if self.use_sim_time:
            self.poseSubscription = self.create_subscription(Odometry,
                                                                 'odom',
                                                                 self.pose_callback,
                                                                 1,
                                                                 callback_group=self.group)
            self.poseSubscription  # prevent unused variable warning
        else:
            self.poseSubscription = self.create_subscription(PoseWithCovarianceStamped,
                                                                 'amcl_pose',
                                                                 self.pose_callback,
                                                                 1,
                                                                 callback_group=self.group)
            self.poseSubscription  # prevent unused variable warning


        self.subscriberManager_timer = self.create_timer(2, self.subscriberManager_callback, callback_group=self.group)
        self.plannedActions_subscriptions = {}
        self.knownPlannedPaths = {}

        self.get_logger().info('Created subscriptions')

        self.plannedActionsPublisher_ = self.create_publisher(PlannedActions, 'plannedActions', 1, callback_group=self.group)

        if self.rviz == True:
            self.plannedPathPublisher_rviz = {}
            for tau in range(self.Tau_span):
                #self.plannedPathPublisher_rviz[tau] = self.create_publisher(PoseWithCovarianceStamped, 'Path_rviz_'+str(tau), 1, callback_group=self.group)
                self.plannedPathPublisher_rviz[tau] = self.create_publisher(PointCloud, 'Path_PointCloud_'+str(tau), 1, callback_group=self.group)

        self.goalPublisher_ = self.create_publisher(PointStamped, 'currentGoal', 1, callback_group=self.group)
        goalMsg = PointStamped()
        if self.use_sim_time:
            goalMsg.header.frame_id = "odom"
        else:
            goalMsg.header.frame_id = "map" 
        goalMsg.header.stamp = self.get_clock().now().to_msg()
        goalMsg.point.x = self.z_plan[self.i_plan][0].item()
        goalMsg.point.y = self.z_plan[self.i_plan][1].item()
        goalMsg.point.z = 0.0
        self.goalPublisher_.publish(goalMsg)

        self.velPublisher_ = self.create_publisher(Twist, 'publishedVel', 1)

        self.get_logger().info('Created publishers')

        planning_timer_period = self.deltaT.item()  # seconds
        self.planning_timer = self.create_timer(planning_timer_period, self.planning_callback, callback_group=self.group)

        # align planning_timer for all robots
        self.t0 = alignTimer(self.get_clock(),self.planning_timer)

        self.tau, mod = divmod(self.t0.nanoseconds, int(self.deltaT.item()*(10**9)))
        if mod > self.deltaT.item()*(10**9)/2: # divmod rounds down
            self.tau = self.tau + 1

        self.get_logger().info('Controller Initialized')
        self.get_logger().info('z_0' + str(self.z_meas))

        if self.Log_DATA:
            outdir = '/root/DATA/logs/' + self.ROBOTNAMESPACE
            if not os.path.exists(outdir):
                os.mkdir(outdir)
 
            filename = self.ROBOTNAMESPACE + "-" + datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".pkl"
            logFilePath = os.path.join(outdir, filename) 
            self.logFileObject = open(logFilePath, 'ab')
            self.get_logger().info('Data Logging Initialized')

    def subscriberManager_callback(self):
        topic_to_find = "plannedActions"

        topics = self.get_topic_names_and_types()

        for topic in topics:
            if topic_to_find in topic[0] and topic[0] != self.plannedActionsPublisher_.topic_name: #[0] for the name as a string
                allready_subscriped = False
                for ID in self.plannedActions_subscriptions: # check if a subscription already exists
                    if topic[0] == self.plannedActions_subscriptions[ID].topic_name:
                        allready_subscriped = True
                        break # a subscription already exists, no need to look further

                if allready_subscriped == False: # if no subscription existed then create one
                    ID = topic[0]
                    ID = ID.replace("/controller/plannedActions","")
                    ID = ID.replace("/","")  
                    self.plannedActions_subscriptions[ID] = self.create_subscription(PlannedActions,
                                                                                     topic[0],
                                                                                     partial(self.plannedActionsSubcriber_callback, topic, ID), # partial is used to get addtional arguments for the call back
                                                                                     1) # add a subscription...

                    self.get_logger().info('Created subscription for topic "' + topic[0] + '"' + ' and assigned it the ID "' + ID + '"')

        # TODO: add feature to delete subscriptions again

    def unpackPoseWithCovarianceMsg(self,poseWithCovariance):
        x = poseWithCovariance.pose.position.x
        y = poseWithCovariance.pose.position.y
        z = poseWithCovariance.pose.position.z

        orientation_q = poseWithCovariance.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        #roll, pitch, yaw = self.quaternionMsg2RPY(orientation_q)

        z_tmp = torch.tensor([x, y, z, roll, pitch, yaw],dtype=torch.float)
        cov_row_major = torch.tensor([poseWithCovariance.covariance],dtype=torch.float) # in row major format

        indices = torch.tensor([0,1,5]) # we want to extract the marginal for x, y, yaw

        mean_marginal, cov_marginal = getNormalMarginal(z_tmp, cov_row_major, indices)

        # zero diagonal vector is assumed to be an error and should only happen in the beginning. Thus, a small number is added!
        #cov_marginal = cov_marginal + torch.diag(0.001*torch.ones(cov_marginal.shape[0]))

        # the covariances from AMCL seems very conservative - probably due to a need for population error in the particle filter - thus we scale it to be less conservative
        cov_marginal = cov_marginal*self.covariance_scale
        mean = mean_marginal
        L = torch.cholesky(cov_marginal)

        return mean, L
    
    def plannedActionsSubcriber_callback(self, topic, ID, msg):

        if ID not in self.knownPlannedPaths:
            self.knownPlannedPaths[ID] = {} # create empty dict

        self.knownPlannedPaths[ID]["radius"] = msg.radius
        self.knownPlannedPaths[ID]["planning_time"] = torch.tensor([msg.planning_time],dtype=torch.float)
        self.knownPlannedPaths[ID]["time_left_from_predicted_pose"] = torch.tensor([msg.time_left_from_predicted_pose],dtype=torch.float)
        poseWithCovariance = msg.pose.pose
        mean, L = self.unpackPoseWithCovarianceMsg(poseWithCovariance)
        poseDistribution = torch.distributions.MultivariateNormal(mean, scale_tril=L)
        self.knownPlannedPaths[ID]["z_meas_dist"] = poseDistribution
        self.knownPlannedPaths[ID]["a_current"] = torch.FloatTensor(msg.a_current)
        self.knownPlannedPaths[ID]["a_alpha"] = torch.reshape(torch.FloatTensor(msg.a_alpha), (self.Tau_span, 2))
        self.knownPlannedPaths[ID]["a_beta"] = torch.reshape(torch.FloatTensor(msg.a_beta), (self.Tau_span, 2))
        self.get_logger().info('Recieved planned actions from: ' + ID)

    def pose_callback(self, msg):
        if not self.use_sim_time:
            z_meas_tmp, self.meas_L = self.unpackPoseWithCovarianceMsg(msg.pose)
        else:
            self.z_meas, self.meas_L = self.unpackPoseWithCovarianceMsg(msg.pose)

    def packPlannedActionsMsg(self, z_meas_tau, z_L_tau, a_current, a_alpha, a_beta, planning_time, time_left_from_predicted_pose):
        msgPlannedActions = PlannedActions()
        time_stamp_msg = self.get_clock().now()
        current_time = time_stamp_msg.to_msg()
        msgPlannedActions.header.stamp = current_time
        if self.use_sim_time:
            msgPlannedActions.header.frame_id = "odom"
        else:
            msgPlannedActions.header.frame_id = "map"

        msgPlannedActions.radius = self.radius
        msgPlannedActions.planning_time = planning_time.item()
        msgPlannedActions.time_left_from_predicted_pose = time_left_from_predicted_pose.item()

        # Current pose estimate
        poseWithCovariance = PoseWithCovarianceStamped()
        if self.use_sim_time:
            poseWithCovariance.header.frame_id = "odom"
        else:
            poseWithCovariance.header.frame_id = "map"
        poseWithCovariance.header.stamp = current_time
        poseWithCovariance.pose.pose.position.x = z_meas_tau[0].item()
        poseWithCovariance.pose.pose.position.y = z_meas_tau[1].item()

        q = quaternion_from_euler(0.0, 0.0, z_meas_tau[2].item())
        poseWithCovariance.pose.pose.orientation.x = q[0]
        poseWithCovariance.pose.pose.orientation.y = q[1]
        poseWithCovariance.pose.pose.orientation.z = q[2]
        poseWithCovariance.pose.pose.orientation.w = q[3]


        newIndices = torch.tensor([0,1,5])
        newShape = torch.tensor([6,6])
        z_cov_tau = z_L_tau*torch.transpose(z_L_tau, 0, 1) # cov = L*L^T
        poseWithCovariance.pose.covariance = expandCovarianceMatrix(z_L_tau, newShape, newIndices).flatten().tolist()
        msgPlannedActions.pose = poseWithCovariance

        # planned actions
        msgPlannedActions.a_current = a_current.flatten().tolist()
        msgPlannedActions.a_alpha = a_alpha.flatten().tolist()
        msgPlannedActions.a_beta = a_beta.flatten().tolist()

        return msgPlannedActions

    def publishPathPointCloud(self, pathPoints):
        time_stamp_path = Time(nanoseconds=(self.tau*int(self.deltaT*(10**9))))

        for tau in range(self.Tau_span):
            dur = Duration(nanoseconds=(tau+1)*int(self.deltaT*(10**9)))
            time_stamp_future = time_stamp_path.__add__(dur)
            pointCloud_tmp = PointCloud()
            pointCloud_tmp.header.stamp = time_stamp_future.to_msg()
            if self.use_sim_time:
                pointCloud_tmp.header.frame_id = "odom"
            else:
                pointCloud_tmp.header.frame_id = "map"

            for i in range(pathPoints.shape[0]):
                point_tmp = Point32()
                point_tmp.x = pathPoints[i,tau,0].item()
                point_tmp.y = pathPoints[i,tau,1].item()
                point_tmp.z = 0.0
                pointCloud_tmp.points.append(point_tmp)

            self.plannedPathPublisher_rviz[tau].publish(pointCloud_tmp)


    def getPoseFromTF(self):
        source_frame = 'map'
        target_frame = 'base_link'

        transformation_timestamped = self.tfbuffer.lookup_transform(source_frame, target_frame, rclpy.time.Time(seconds=0))
        x = transformation_timestamped.transform.translation.x
        y = transformation_timestamped.transform.translation.y

        orientation_q = transformation_timestamped.transform.rotation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)


        z_meas = torch.tensor([x, y, yaw],dtype=torch.float)

        return z_meas

    def planning_callback(self):
        self.tau = self.tau + 1

        if not self.use_sim_time:
            try:
                self.z_meas = self.getPoseFromTF()
            except Exception as e:
                self.get_logger().info('Error: ' + str(e))


        counter = 0
        # Check if current goal is reached
        if torch.dist(self.z_meas[[0,1]],self.z_plan[self.i_plan].index_select(0, torch.tensor([0, 1])),p=2) < self.z_plan[self.i_plan][torch.tensor([3])]:
            if self.i_plan+1 == self.z_plan.shape[0]:
                self.i_plan = 0
            else:
                self.i_plan = self.i_plan+1
            self.get_logger().info('New goal: ' + str(self.z_plan[self.i_plan]))
            if self.rviz == True:
                goalMsg = PointStamped()
                if self.use_sim_time:
                    goalMsg.header.frame_id = "odom"
                else:
                    goalMsg.header.frame_id = "map"
                goalMsg.header.stamp = self.get_clock().now().to_msg()
                goalMsg.point.x = self.z_plan[self.i_plan][0].item()
                goalMsg.point.y = self.z_plan[self.i_plan][1].item()
                goalMsg.point.z = 0.0
                self.goalPublisher_.publish(goalMsg)

        a_alpha_tau = torch.tensor([100., 1000.],dtype=torch.float) # small preference for going forward initially 
        a_beta_tau = torch.tensor([10., 1000.],dtype=torch.float)

        # Plan next action(s)
        while self.est_planning_epoch_time.__lt__(Duration(nanoseconds=0.80*self.planning_timer.time_until_next_call())): # check if there is sufficient time to run again
            tic_true = time.time()

            tic = self.get_clock().now()
            if not self.use_sim_time:
                try:
                    self.z_meas = self.getPoseFromTF()
                except Exception as e:
                    self.get_logger().info('Error: ' + str(e))
            
            z_meas_tau = self.z_meas 
            z_L_tau = self.meas_L
            knownPlannedPaths_tau_i = self.knownPlannedPaths # does a good job at avoiding concurrent access but not entirely safe..

            # calculate dt to next timestep:
            time_left = torch.tensor([self.planning_timer.time_until_next_call()/(10**9)],dtype=torch.float)# s
            planning_time = torch.tensor([self.est_planning_epoch_time.nanoseconds/(10**9)],dtype=torch.float)# s
            time_left_from_predicted_pose = time_left - planning_time
            z_current_dist = dist.MultivariateNormal(z_meas_tau, scale_tril=z_L_tau)

            # Run SVI
            losses = []
            for svi_epoch in range(self.svi_epochs):
                losses.append(self.svi.step(self, z_current_dist, self.a_current, self.deltaT, self.z_plan[self.i_plan], knownPlannedPaths_tau_i, planning_time, time_left_from_predicted_pose, T1 = self.tau, T2 = self.tau+self.Tau_span))

            # Send msg to other robots
            a_alpha = torch.zeros(self.Tau_span,2)
            a_beta = torch.zeros(self.Tau_span,2)
            for Tau in range(self.tau, self.tau+self.Tau_span):
                a_alpha[Tau-self.tau,:] = pyro.param("a_alpha_{}".format(Tau)).detach()
                a_beta[Tau-self.tau,:] = pyro.param("a_beta_{}".format(Tau)).detach()

            msgPlannedActions = self.packPlannedActionsMsg(z_meas_tau, z_L_tau, self.a_current, a_alpha, a_beta, planning_time, time_left_from_predicted_pose)
            self.plannedActionsPublisher_.publish(msgPlannedActions)

            if self.rviz == True:
                # Generate samples of predicted path:
                N = 100
                z_tmp = torch.zeros(N,self.Tau_span,3)
                for i in range(N): 
                    z_tmp[i,:,:] = motionModel(z_current_dist, self.a_current, a_alpha, a_beta, self.deltaT, planning_time, time_left_from_predicted_pose, T1 = self.tau, T2 = self.tau+self.Tau_span)

                self.publishPathPointCloud(z_tmp)

            if self.Log_DATA:
                # data to save
                logDict = {
                            "timestamp": tic_true, # int
                            "tau": self.tau, # int
                            "counter": counter, # int
                            "losses": losses, # [] list
                            "z_meas_tau": z_meas_tau, # torch.tensor([0,0,0],dtype=torch.float)
                            "z_L": z_L_tau, # z_cov = torch.diag(torch.tensor([1.,1.,1.],dtype=torch.float))
                            "a_current": self.a_current, # torch.tensor([0., 0.],dtype=torch.float)
                            "a_alpha": a_alpha, # torch.zeros(self.Tau_span,2)
                            "a_beta": a_beta, # torch.zeros(self.Tau_span,2)
                            "z_tmp": z_tmp, # torch.zeros(N,self.Tau_span,3)
                            "currentGoal": self.z_plan[self.i_plan],
                            "tau_span": self.Tau_span,
                            "r_plan": self.r_plan, 
                            "radius": self.radius
                }
                pickle.dump(logDict, self.logFileObject)

            # Draw actions
            # Get action distribution parameters
            a_alpha_tau = pyro.param("a_alpha_{}".format(self.tau)).detach()
            a_beta_tau = pyro.param("a_beta_{}".format(self.tau)).detach()

            # Sample random action from p(a_tau|O=1,C=0)
            a_tau = torch.distributions.Beta(a_alpha_tau, a_beta_tau).rsample().detach()

            # Alternatively, choose mean action...
            a_tau[0] = a_alpha_tau[0]/(a_alpha_tau[0]+a_beta_tau[0])
            a_tau[1] = a_alpha_tau[1]/(a_alpha_tau[1]+a_beta_tau[1])

            self.a_next = a_tau # save current estimate of next action to take

            self.a_current = self.a_next
            a = action_transforme(self.a_current) # scale actions appropriately

            msgVel = Twist()
            msgVel.linear.x = a[0].item()
            msgVel.angular.z = a[1].item()
            self.velPublisher_.publish(msgVel)

            # estimate execution time by Cumulative moving average
            toc = self.get_clock().now()
            toc_true = time.time() - tic_true
            dur = toc.__sub__(tic).nanoseconds
            if not (dur >= int(self.deltaT*(10**9))):
                dur_curr_est = self.est_planning_epoch_time.nanoseconds
                dur_new_est = (self.N_samples_epoch_time *dur_curr_est+dur) / (self.N_samples_epoch_time + 1) # Cumulative moving average
                self.est_planning_epoch_time = Duration(nanoseconds=dur_new_est)
                self.N_samples_epoch_time = self.N_samples_epoch_time + 1

            counter = counter + 1

        self.get_logger().info('N iterations: ' + str(counter) + '  est_planning_epoch_time: ' + str(self.est_planning_epoch_time.nanoseconds/10**6) + "    goal: " + str(self.z_plan[self.i_plan]) + " x: " + str(self.z_meas))
        time_left = self.planning_timer.time_until_next_call()
        if time_left < 0:
            self.get_logger().warn("Planning did not finish in time - spend " + str(-1.*time_left/10**6) + " ms to much")

def main(args=None):
    rclpy.init(args=args)

    trajectory_planner = TrajectoryPlanner()

    executor = MultiThreadedExecutor(num_threads=2) # when simulating we need two threads for the time to be updates i.e. when using "self.get_clock().now()"
    executor.add_node(trajectory_planner)
    executor.spin()

    # Destroy the node explicitly
    trajectory_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
