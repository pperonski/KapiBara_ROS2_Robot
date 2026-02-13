#!/usr/bin/env python3
import codon

import os
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from sensor_msgs.msg import Range,Imu,PointCloud2
from geometry_msgs.msg import Quaternion

from ament_index_python.packages import get_package_share_directory,get_package_prefix

from std_msgs.msg import Float64MultiArray

from kapibara.DeepIDTFLite import DeepIDTFLite

from kapibara_interfaces.msg import Emotions
from kapibara_interfaces.msg import Microphone
from kapibara_interfaces.msg import PiezoSense

from sensor_msgs.msg import CompressedImage,Image
from sensor_msgs.msg import PointCloud2,PointField

from sensor_msgs.msg import CompressedImage,Image
from sensor_msgs.msg import PointCloud2,PointField

from rcl_interfaces.msg import ParameterDescriptor

from kapibara_interfaces.msg import FaceEmbed

from kapibara_interfaces.srv import StopMind

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

import webrtcvad

import cv2
from cv_bridge import CvBridge

import base64
import numpy as np

from timeit import default_timer as timer
import time

import librosa

from qdrant_client.models import VectorParams, Distance,OrderBy,PayloadSchemaType
from qdrant_client.http import models

import uuid

from tiny_vectordb import VectorDatabase
from kapibara.kapi_graphs import Graph

'''

A working of map mind of KapiBara robot is inner working is based 
on graph databases:

- Graph database that represents relations between states and actions connecting them
- Emotions relations ship that to every detected object it associates emotional state, when it is diffrent than 0

Our robot will move to the place with more positive emotional state and use few estimators that will 
attach those points to the map, for example when our robots see a object that makes him happy it 
will use readings from point cloud to estimate its position in space and then add point with positive emotional state to the map.

We can use something similar for sound, too, and then estimate a direction of sound source and add its source.

Action graph works on principle when a state in the graph is triggered it will search through actions to find target state 
with most positive outcome and then perform actions collected on the path to that state.


'''


ID_TO_EMOTION_NAME = [
    "angry",
    "fear",
    "happiness",
    "uncertainty",
    "boredom"
    ]

IMG_EMBEDDING_SHAPE = 32*32*4
ML_SPECTOGRAM_EMBEDDING_SHAPE = 32*32

# month, day, hour, minute, second
TIME_EMBEDDING_SHAPE = 5

OBJECT_DB = "object_database"
OBJECT_EMBEDDING_SHAPE = 32*32

POINT_MAP_DB = "points_database"
POINT_IMG_DB = "points_imgages"
POINT_SPEC_DB = "points_spect"
POINT_TIME_DB = "points_time"


FACE_TOLERANCE = 0.94

import chromadb

class MapMind(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('tick_time', 0.05)
        
        package_path = get_package_share_directory('kapibara')
        
        self.declare_parameter('yolo_model_path','yolov11n-face_float32.tflite')
        self.declare_parameter('deepid_model_path','deepid.tflite')
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        
        self.face_yolo = YOLO(os.path.join(package_path,'model',yolo_model_path))
        
        deepid_model_path = self.get_parameter('deepid_model_path').get_parameter_value().string_value
        
        self.deep_id = DeepIDTFLite(filepath=os.path.join(package_path,'model',deepid_model_path))
        
        self.twist_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.emotion_pub = self.create_publisher(Emotions,'emotions', 10)
        
        self.spectogram_publisher = self.create_publisher(Image, 'spectogram', 10)
        
        self.timer_period = self.get_parameter('tick_time').get_parameter_value().double_value  # seconds
        self.timer = self.create_timer(self.timer_period, self.tick_callback)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        self.image_sub = self.create_subscription(
            Image,
            'camera',
            self.image_callback,
            10)
        
        self.image_sub = self.create_subscription(
            Image,
            'depth',
            self.depth_image_callback,
            10)
        
        self.points_sub = self.create_subscription(
            PointCloud2,
            'points',
            self.points_callback,
            10)
        
        self.ext_emotion_sub = self.create_subscription(
            Emotions,
            'ext_emotion',
            self.ext_emotion_callback,
            10)
        
        self.mic_subscripe = self.create_subscription(Microphone,'mic',self.mic_callback,10)
                
        self.bridge = CvBridge()
        
        self.current_face_embeddings = []
        
        self.initialize_db()
        
        # Embedings have a form of 64x64 image/spectogram data
        self.action_graph = Graph(database_name="actions_graph")
        
        # A list of objects found in the scene, 64x64 embeddings with bouding boxes
        self.found_objects = []
        
        # A map of points with X,Y positions and associated emotional state, and
        # decay factor
        self.point_map = []
        
        # self.face_db = FaceSqlite(self.db)
        
        self.pat_detected = 0.0
                
        self.emotion_state = 0.0
        
        self.last_img = np.zeros((64,64),dtype=np.float32)
        self.last_spectogram = np.zeros((64,64),dtype=np.float32)
        
        self.action_list = []
        self.action_executing = False
        self.action_iter = 0
        
        self.image = None
        self.depth = None
        self.spectogram = None
        
        self.last_cmd = (0.0,0.0)
        
        self.position = np.zeros(3)
        
        self.emotion = Emotions()
        
        # emotion state from external sources
        self.ext_emotion = Emotions()
        
        self.obstacle_detected = False
        
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        
        self.audio_fear = 0
        
        self.mic_buffor = np.zeros(2*16000,dtype=np.float32)
        
        self.wait_for_img = True
        self.wait_for_depth = True
        
        self.last_score = 0
        
        self.last_id_spec = ""
        self.last_id_img = ""
        
        # robot position in the map
        self.x = 0
        self.y = 0
        self.yaw = 0
        
        # angry
        # fear
        # happiness
        # uncertainty
        # boredom 
        # anguler values for each emotions state
        self.emotions_angle=[0.0,180.0,25.0,145.0,90.0] 
        
        self.ears_publisher = self.create_publisher(
            Float64MultiArray,
            'ears_controller/commands', 
            10
            )
        
    def reinitialize_db(self):
        
        self.initialize_db()
        
    def initialize_db(self):
        
        self.associated_db = chromadb.PersistentClient("association_db")
        self.obj_db = self.associated_db.create_collection("obj",get_or_create=True)
        
    def send_command(self,linear:float,angular:float):
        
        twist = Twist()
        
        linear = min(linear,1.0)
        linear = max(linear,-1.0)
        
        angular = min(angular,1.0)
        angular = max(angular,-1.0)
        
        twist.linear.x = linear*self.max_linear_speed
        twist.angular.z = angular*self.max_angular_speed
        
        self.twist_pub.publish(twist)
        
        # estimate position based on speed solely
        # it will be filter out using external odometry /
        # map info
        self.yaw += angular*self.timer_period
        
        self.x += linear*np.cos(self.yaw)*self.timer_period
        self.y += linear*np.sin(self.yaw)*self.timer_period
        
    def stop(self):
        self.send_command(0.0,0.0)
        
    def odom_callback(self,odom:Odometry):
        self.position = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
            ])
    
    def mic_callback(self,mic:Microphone):
        
        self.get_logger().debug("Mic callback")
        # I have to think about it
        
        self.mic_buffor = np.roll(self.mic_buffor,mic.buffor_size)
        
        left = np.array(mic.channel1,dtype=np.float32)/np.iinfo(np.int32).max
        right = np.array(mic.channel2,dtype=np.float32)/np.iinfo(np.int32).max
        
        combine = ( left + right ) / 2.0
        
        self.mic_buffor[:mic.buffor_size] = combine[:]
        
        start = timer()
        
        spectogram = librosa.feature.melspectrogram(y=self.mic_buffor, sr=16000,n_mels=224,hop_length=143)
        
        self.get_logger().info(f"Spectogram size: {spectogram.shape}")
        
        # publish last spectogram
        self.spectogram_publisher.publish(self.bridge.cv2_to_imgmsg(spectogram))
                
        self.get_logger().debug("Hearing time: "+str(timer() - start)+" s")
        
        self.spectogram = cv2.resize(spectogram,(64,64),interpolation=cv2.INTER_LINEAR) / 255.0
        
        # Indicate that it is audio data
        self.spectogram[0] = -10.0
        
        mean = np.mean(np.abs(combine))
        
        if mean >= 0.7:
            self.audio_fear = 1.0
            
    def ext_emotion_callback(self,msg:Emotions):
        self.ext_emotion = msg
    
    def depth_image_callback(self,msg:Image):
        
        self.get_logger().info('I got depth image with format: %s' % msg.encoding)
        
        self.depth = self.bridge.imgmsg_to_cv2(msg)
        
        self.wait_for_depth = False
    
    def image_callback(self,msg:Image):
        
        self.get_logger().info('I got image with format: %s' % msg.encoding)
        
        self.image = self.bridge.imgmsg_to_cv2(msg)
        
        self.wait_for_img = False
    
    @codon.jit
    @staticmethod
    def points_callback_codon_code(points:np.ndarray):
        sorted_points = np.sort(points)
        
        min_points = np.mean(sorted_points[0:10])
        
        obstacle_detected = float(np.exp(-min_points*25))
        
        obstacle_detected = min(obstacle_detected,1.0)
        
        if obstacle_detected < 0.01:
            obstacle_detected = 0.0
            
        return obstacle_detected
      
    def points_callback(self, msg: PointCloud2):
        
        start = timer()
        
        # Read points from PointCloud2
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
            reshape_organized_cloud=True
        )
        
        self.obstacle_detected = MapMind.points_callback_codon_code(points)
                
        if self.obstacle_detected:    
            self.get_logger().info(f'Obstacle detected! {self.obstacle_detected}, time: {timer() - start} s')
            
    def sense_callabck(self,sense:PiezoSense):
        
        # should be rewritten 
        
        pin_states = sense.pin_state
        
        patting_sense = pin_states[4]
                
        if patting_sense:
            
            self.get_logger().debug('Patting detected')
                        
            self.pat_detected = 1.0
                        
            score = 10
                
    @codon.jit            
    def emotion_state_calculate(self,emotions:list[float]):
                
        return  emotions[2]*320.0 + emotions[1]*-120.0 + emotions[3]*-40.0 + emotions[0]*-60.0 + emotions[4]*-20.0
    
    def send_ears_state(self,emotions:list[float]):
        
        max_id = 4
        
        if np.sum(emotions) >= 0.01:
            max_id = np.argmax(emotions[:4])
            
        self.get_logger().debug("Current emotion: "+str(ID_TO_EMOTION_NAME[max_id]))
            
        self.get_logger().debug("Sending angle to ears: "+str(self.emotions_angle[max_id]))
        
        angle:float = (self.emotions_angle[max_id]/180.0)*np.pi
        
        array=Float64MultiArray()
        
        array.data=[np.pi - angle, angle]
        
        self.ears_publisher.publish(array)
        
    
    def check_actions_payload(self,payload:dict):
        
        return "actions" in payload.keys() \
                and "score" in payload.keys() \
                and type(payload["actions"]) is list \
                and type(payload["score"]) is float
                
    def add_score_to_embeddings(self,score:float,embedding:np.ndarray):
        """
        Docstring for add_score_to_embeddings
        
        :param self: Description
        :param score: float value to be associated with the embedding
        :param embedding: numpy array representing the embedding to be stored
                v - associated value
        :type points: list[tuple]
        """
                
        new_id = uuid.uuid4()
                                        
        payload = str(score)
        
        self.get_logger().debug("Updating image embedding.")
        
        points_id = [str(new_id)+":"+payload]
        
        self.general_db.setBlock(
            points_id,
            [embedding]
        )
    
    def get_score_from_embeddings(self,embedding:np.ndarray)->float:
        # check if embeddings aren't aleardy presents
        
        search_ids, search_scores = self.general_db.search(embedding, k=4)
        
        for id,score in zip(search_ids, search_scores):
            if score > 0.1:
                continue
                
            payload = id.split(":")[1].encode('ascii')
            
            return float(payload)
        
    def move_towards_point(self):
        """
        Returns command that will move robot 
        closer to the most positive point
        """
        # TODO
        return (0,0)
    
    def point_seeking(self,score:float):
        # Update embeddings in the database with new score
        
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_LINEAR) / 255.0
        
        # This way we indicate that it is image data
        img[0] = 10.0
        
        pos = (
            self.position[0],
            self.position[1],
            self.position[2],
        )
        
        # Update embeddings with audio data
        img_id = self.action_graph.add_node(img,score)
        
        # Update graph with new score and connections
        spec_id = self.action_graph.add_node(self.spectogram,score)
        
        # Update graph connections
        if len(self.last_id_img):
            self.action_graph.connect_nodes(self.last_id_img,img_id,self.last_cmd)
        if len(self.last_id_spec):
            self.action_graph.connect_nodes(self.last_id_spec,spec_id,self.last_cmd)
            
        self.last_id_img = img_id
        self.last_id_spec = spec_id
        
        # propagate score, through graph
        self.propogate_value_through_graph(spec_id,score)
        self.propogate_value_through_graph(img_id,score)
        
        # Retrive points
        # Retrive points from visible objects
        # TODO
        
        for point in self.point_map:
            point[3] = point[3]*0.95
        
        # Remove points that have low lifetime
        self.point_map = list(filter(lambda x: x[3] > 0.01,self.point_map))              
        
        # They are not needed anymore
        self.found_objects.clear()
        
        # Move towards point with higher positive value
        # Here is a algorithm for moving
        cmd = self.move_towards_point()
        
        self.send_command(cmd[0],cmd[1])
        
        self.last_cmd = cmd
        
        dscore = score - self.last_score

        self.last_score = score
        
        self.last_spectogram = self.spectogram
        self.last_img = img
        
        if score >= 0.0:
            self.stop()
            return
        
            
        img_action = self.get_commands_for_state(img_id)
        
        spec_action = self.get_commands_for_state(spec_id)
        
        if img_action is None and spec_action is None:
            return
        
        if img_action is None and spec_action is not None:
            self.send_command(spec_action[1],spec_action[2])
            return
        
        if img_action is not None and spec_action is None:
            self.send_command(img_action[1],img_action[2])
            return
        
        if img_action[0] > spec_action[0]:
            self.send_command(img_action[1],img_action[2])
        else:
            self.send_command(spec_action[1],spec_action[2])
            
                                    
    def get_commands_for_state(self,node_id:str):
        conns = self.action_graph.get_connections(node_id)
        
        if len(conns) == 0:
            return None
        
        nodes = self.action_graph.get_node_by_ids(conns)
        
        values = [(node[1],conn[1],conn[2]) for node,conn in zip(nodes,conns)]
        
        best_action = max(values,key=lambda x: x[0])
        
        return best_action
        
        
    
    def propogate_value_through_graph(self,node_id:str,value:float,depth:int=10):
        
        if depth == 0:
            return
        
        conns = self.action_graph.get_backwards_connections(node_id)
        
        _value = 0.8*value
        _depth = depth - 1
        
        _from_ids = [ conn[0] for conn in conns]
        
        nodes = self.action_graph.get_node_by_ids(_from_ids)
        
        values = [node[1]*0.6 + 0.4*_value for node in nodes]
        
        self.action_graph.update_nodes_metadata(_from_ids,values)
        
        for id in _from_ids:                        
            self.propogate_value_through_graph(id,_value,_depth)
        
    def action_execution(self,score:float):
        
        dscore = score - self.last_score

        self.last_score = score
        
        cmd = self.action_list[self.action_iter]
        
        self.send_command(cmd)
        
        if self.action_iter < len(self.action_list):
            self.action_iter += 1
        else:
            self.action_executing = False
            return
        
        if dscore < -1.0:
            self.action_executing = False
                
    def tick_func(self):
        # emotion validation pipeline
        
        # face detection
        
        self.current_face_embeddings.clear()
        
        # evaluate emotion states
        
        emotions = Emotions()
        
        self.emotion.happiness = self.pat_detected*10.0
        self.emotion.fear = self.obstacle_detected*1.0
        
        self.pat_detected = 0.0
        
        emotions_arr = [
            self.emotion.angry + self.ext_emotion.angry,
            self.emotion.fear + self.ext_emotion.fear,
            self.emotion.happiness + self.ext_emotion.happiness,
            self.emotion.uncertainty + self.ext_emotion.uncertainty,
            self.emotion.boredom + self.ext_emotion.boredom
        ]
        
        emotions.angry = emotions_arr[0]
        emotions.fear = emotions_arr[1]
        emotions.happiness = emotions_arr[2]
        emotions.uncertainty = emotions_arr[3]
        emotions.boredom = emotions_arr[4]
        
        self.emotion_pub.publish(emotions)
        
        score = self.emotion_state_calculate(emotions_arr)
                
        # send ears position
        self.send_ears_state(emotions_arr)
        
        self.point_seeking(score)
        
        # Think about inner workings of boredom
        # Move to the direction of point with higher score
        
        
    def tick_callback(self):
        
        if self.wait_for_img or self.wait_for_depth or self.spectogram is None:
            return
        
        self.timer.cancel()
        
        self.get_logger().info('Mind tick')
        
        start = timer()
        
        self.tick_func()
        
        end = timer()
        
        self.get_logger().info(f'Tick inference time {end - start} s')
        
        self.timer.reset()
        

def main(args=None):
    rclpy.init(args=args)

    map_mind = MapMind()

    rclpy.spin(map_mind)

    map_mind.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()