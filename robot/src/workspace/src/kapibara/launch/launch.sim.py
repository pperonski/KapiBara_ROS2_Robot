import os
import logging
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription,TimerAction,RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

from launch_ros.actions import Node
import xacro

from launch.actions import SetEnvironmentVariable

def generate_launch_description():

    # Specify the name of the package and path to xacro file within the package
    pkg_name = 'kapibara'
    file_subpath = 'description/kapibara.urdf.xacro'


    # Use xacro to process the file
    xacro_file = os.path.join(get_package_share_directory(pkg_name),file_subpath)
    robot_description_raw = xacro.process_file(xacro_file,mappings={'sim_mode' : 'true','robot_name' : 'KapiBara'}).toxml()

    gazebo_env = SetEnvironmentVariable("GAZEBO_MODEL_PATH", os.path.join(get_package_prefix("kapibara"), "share"))
    
    # Configure the node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace = 'KapiBara',
        parameters=[{'robot_description': robot_description_raw,
        'use_sim_time': True}] # add other parameters here if required
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py',)]),
            launch_arguments={
                'world': '/app/src/rviz/small_house.world',
                'params_file': os.path.join(get_package_share_directory(pkg_name),"config/gazebo.yaml"),
                }.items()
        )

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                    arguments=["-topic","/KapiBara/robot_description","-entity","kapibara","-timeout","240","-z","0.0","-y","-4.5","-x","-3.5"],
                    output='screen')
    
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace="KapiBara",
        arguments=["motors",'--controller-manager-timeout','240'],
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace="KapiBara",
        arguments=["joint_broad",'--controller-manager-timeout','240'],
    )
    
    ears_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace="KapiBara",
        arguments=["ears_controller",'--controller-manager-timeout','240'],
    )

    emotions = Node(
        package="emotion_estimer",
        executable="estimator.py",
        namespace="KapiBara",
        parameters=[{
            "sim":True
        }]
    )
    
    mic = Node(
        package="microphone",
        executable="mic.py",
        namespace = 'KapiBara',
        arguments=[],
        parameters=[{"channels":2,"sample_rate":44100,"chunk_size":4096,"device_id":5}],
        output='screen'
    )
    
    map_mind = Node(
        package="kapibara",
        executable="map_mind.py",
        namespace="KapiBara",
        arguments=[],
        parameters=[{
            'use_sim_time': True,
        }],
        remappings=[
            ('cmd_vel','motors/cmd_vel_unstamped'),
            ('camera','camera/image_raw'),
            ('depth','camera/depth/image_raw'),
            ('points','camera/points')
        ]
    )
    
    mola_cfg = os.path.join(get_package_share_directory(pkg_name), 'config', 'mola_cfg.yaml')
    
    tf_remaps = [('tf', '/tf'),
                 ('tf_static', '/tf_static')]

    
    mola = Node(
            package='mola_launcher',
            executable='mola-cli',
            output='screen',
            parameters=[{'use_sim_time': True}],
            remappings=tf_remaps,
            namespace="KapiBara",
            arguments=[mola_cfg],
        )
    
    delayed_mola= TimerAction(
       actions=[
            mola
           ],
        period=10.0
    )
    
    # Run the node
    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        #rviz,
        spawn,
        # emotions,
        diff_drive_spawner,
        joint_broad_spawner,
        ears_controller_spawner,
        mic,
        map_mind,
        # delayed_mola
        # delayed_rtabmap
    ])


