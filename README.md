Object Detector on NVidia Jetson with ROS
=========================================

## Jetson Setup

We bought the Nvidia Jetson TX2, [Orbitty carrier
board](http://www.wdlsystems.com/Computer-on-Module/Carrier-Boards/CTI-Orbitty-Carrier-for-NVIDIA-Jetson-TX1.html),
and [Connect Tech active heat
sink](http://www.wdlsystems.com/Computer-on-Module/Cooling-Accessories/CTI-NVIDIA-TX1-Module-Active-Thermal-Heatsink.html).
For the AC power adapter, we used a 12 VDC power supply we had sitting around.

Installing JetPack:

* Install Ubuntu 16.04 in a VM and give it enough ram, maybe 4-6 GiB
  ([src](https://devtalk.nvidia.com/default/topic/988616/jetpack-2-3-1-flash-problems-/?offset=7))
* Enable USB3 in VirtualBox (even though the Jetson is only USB2, if you
  only have USB2 you get a "BootRom is not running" error,
  [src](https://devtalk.nvidia.com/default/topic/1002827/jetson-tx2/problem-flashing-tx-2/2))
* Run JetPack installer and install everything on the host.
* When it gets to the point to install to the Jetson, you'll have to put it
  into the Force Recovery mode (unplug power, plug power, hold in on
  RECOVERY, press Reset, let up on Recovery after 2 seconds). Then NVIDIA
  CORP device shows up in "lsusb." In VirtualBox, under USB (bottom right
  corner) check that for the VM. Press Enter in the VM to say to install.
  Then it'll say it doesn't find the USB device. Check the NVIDIA CORP
  again. Then it'll install.
* After copying over the filesystem, it'll error that it can't find the IP.
  Then you can quit the installer.
* Install the Orbitty modifications to support USB and the PWM fan via
  following [their instructions](http://connecttech.com/resource-center/cti-l4t-nvidia-jetson-board-support-package-release-notes/#bkb-h5).
  You'll end up running the command `sudo ./flash.sh orbitty mmcblk0p1`.
  Note, do this *before* installing anything else since this step will
  overwrite the whole filesystem.
* Plug the Jetson into your host computer ethernet (so you can get IP from
  Wireshark) and share your connection or into a router you can get the Jetson
  IP from. Then run the installer again but make sure to select "no action" to
  install the OS ("Flash OS Image to Target") when re-running. Then when it
  gets to installing other software, e.g. CUDA, it'll ask for the IP, user, and
  pass. Specify the IP and then "nvidia" for both username and password.
  ([src](https://devtalk.nvidia.com/default/topic/1002081/jetson-tx2/jetpack-3-0-install-with-a-vm/))
* Note: after it's all done, to shut down VirtualBox, you probably have to
  uncheck the USB device.

Allow LLMNR so we can resolve "tegra-ubuntu" hostname to SSH into it:

    sudo systemctl enable systemd-resolved
    sudo systemctl start systemd-resolved

Disable the display manager (if desired):

    sudo rm /etc/X11/default-display-manager
    sudo touch /etc/X11/default-display-manager

Set CPUs and GPU to max performance on boot (if desired): add this right before
the `exit 0` in */etc/rc.local* script:

    ( sleep 60 && nvpmodel -m 0 && /home/ubuntu/jetson_clocks.sh )&

Enable universe and multiverse repositories (e.g. to install htop):

    sudo add-apt-repository universe
    sudo add-apt-repository multiverse

Update:

    sudo apt update
    sudo apt upgrade

Install TensorFlow

    sudo apt install python{,3}-pip python{,3}-numpy python{,3}-matplotlib htop \
        jnettop protobuf-compiler python{,3}-pil python{,3}-lxml libxml2-dev \
        libxslt1-dev python{,3}-yaml python{,3}-docutils \
        redis-server python{,3}-redis \
        ros-lunar-rosserial ros-lunar-rosserial-arduino

    pip3 install --user virtualenvwrapper
    export WORKON_HOME=~/Envs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source ~/.local/bin/virtualenvwrapper.sh
    echo "export WORKON_HOME=~/Envs" >> ~/.bashrc
    echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
    echo "source ~/.local/bin/virtualenvwrapper.sh" >> ~/.bashrc
    echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc

    # Human detection
    pip2 install --user imutils

    # For Python 3
    mkvirtualenv -p python3 --system-site-packages tf-python3
    workon tf-python3
    git clone https://github.com/jetsonhacks/installTensorFlowJetsonTX.git
    cd installTensorFlowJetsonTX/TX2/
    pip install tensorflow-1.3.0-cp35-cp35m-linux_aarch64.whl
    pip install catkin_pkg rospkg
    deactivate

    # For Python 2
    mkvirtualenv -p python2 --system-site-packages tf-python2
    workon tf-python2
    git clone https://github.com/peterlee0127/tensorflow-tx2.git
    cd tensorflow-tx2
    pip install tensorflow-1.4.1-cp27-cp27mu-linux_aarch64.whl
    deactivate

If using Python 3.5, as
[described](https://devtalk.nvidia.com/default/topic/1027449/jetson-tx2/run-tensorflow-1-3-on-tx2-stuck/post/5226615/)
you need cuDNNv7. Download the .deb from
[https://developer.nvidia.com/nvidia-tensorrt3rc-download](https://developer.nvidia.com/nvidia-tensorrt3rc-download).
Then:

    sudo dpkg -i nv-tensorrt-repo-ubuntu1604-rc-cuda8.0-trt3.0-20170922_3.0.0-1_arm64.deb
    sudo apt update
    sudo apt install tensorrt python3-dev

In your *.ssh/config* for ease of SSHing:

    Host jetson
    HostName tegra-ubuntu
    User nvidia
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes

For TensorFlow Object Detection, add this to the *.bashrc* file:

    export PYTHONPATH=$PYTHONPATH:/home/nvidia/catkin_ws/src/ras-object-detection/models/research/:/home/nvidia/catkin_ws/src/ras-object-detection/models/research/slim/

Copy your model files over to the Jetson *~/networks*:

    scp -r /path/to/ras-object-detection/datasets/SmartHome/*.pb jetson:networks/
    scp /path/to/ras-object-detection/datasets/SmartHome/tf_label_map.pbtxt jetson:networks/
    scp -r /path/to/ras-object-detection/datasets/SmartHome/test_images jetson:networks/

Install ROS ([src](http://wiki.ros.org/lunar/Installation/Ubuntu)):

    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    sudo apt-get update
    sudo apt-get install ros-lunar-desktop-full libroscpp-dev librospack-dev libtf2-ros-dev libtf-dev libnodeletlib-dev
    sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential
    sudo rosdep init
    rosdep update

    source /opt/ros/lunar/setup.bash
    echo "source /opt/ros/lunar/setup.bash" >> ~/.bashrc

Make ROS work with Python 3:

    sudo apt-get install python3-empy # Errors building messages without this

### First Workspace

Create our Catkin workspace:

    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src

Allow working with the Orbbec Astra camera:

    git clone https://github.com/orbbec/ros_astra_launch.git astra_launch
    git clone https://github.com/orbbec/ros_astra_camera.git astra_camera
    git clone https://github.com/ros-drivers/rgbd_launch.git

For working with YOLO:

    git clone --recursive https://github.com/WSU-RAS/darknet_ros.git

Camera Calibration and Depth Sensor:

    git clone https://github.com/ros-perception/image_pipeline

Camera pan/tilt control:

    git clone https://github.com/vanadiumlabs/arbotix_ros.git

Package that allows our code to do coordinate transforms on point clouds.
There's a package "python-tf2-sensor-msgs" that should do this, but it's old
enough that it errors on importing due to some renames.

    sudo apt install ros-lunar-tf2-bullet
    git clone https://github.com/ros/geometry2

The object detection code:

    git clone https://github.com/WSU-RAS/jetson ras_jetson
    git clone https://github.com/WSU-RAS/cob_perception_msgs

For some reason catkin won't build without installing via pip:
    
    pip2 install --user docutils rospkg

Build everything:

    source /opt/ros/lunar/setup.bash
    cd ~/catkin_ws
    catkin_make --pkg darknet_ros_msgs # Needs to be built before darknet_ros
    catkin_make -DFILTER=OFF -DCMAKE_BUILD_TYPE=Release
    catkin_make install

Source this new workspace:

    source ~/catkin_ws/devel/setup.bash
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

Setup udev rules for camera, then make sure to unplug then plug back in the
camera:

    source ~/catkin_ws/devel/setup.bash
    cd ~/catkin_ws/src/astra_camera
    ./scripts/create_udev_rules

Print [checkerboard](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration?action=AttachFile&do=view&target=check-108.pdf)
and measure square in meters. Mine are 0.025 m. Follow the [tutorial](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration).

    rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.025 image:=/camera/rgb/image_raw camera:=/camera/rgb

### Overlay Workspace

Now we need to overlay a new workspace since this repository requires Python 3
which messes up some of what we have in our previous workspace.

Download this repo. Note you may have some issues if you move this after
initializing the submodules.

    mkdir -p ~/catkin_py3/src/
    cd ~/catkin_py3/src/
    git clone --recursive https://github.com/WSU-RAS/jetson-py3.git ras_jetson_py3

Clone the *vision_opencv* package to make *cv_bridge* work with Python 3:

    git clone https://github.com/WSU-RAS/vision_opencv.git

Then, generate the protobuf files:

    cd ~/catkin_py3/src/ras_jetson_py3/models/research/
    protoc object_detection/protos/*.proto --python_out=.

Now build OpenCV 2 for Python 3
([src](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)).
Note you'll need roughly 12 GiB of disk space for this. If you don't have that,
then rsync a bunch of your files off the Jetson, delete them, and then copy
them back after you do the `make install`. Also, make sure you run `sudo
~/jetson_clocks.sh` before doing this so it'll compile faster.

    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev
    wget https://github.com/opencv/opencv/archive/3.3.1.zip -O opencv.zip
    wget https://github.com/opencv/opencv_contrib/archive/3.3.1.zip -O opencv_contrib.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
    cd ~/opencv-3.3.1/
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.1/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
        -D BUILD_EXAMPLES=ON ..
    make -j6
    sudo make install
    sudo ldconfig

Now, if you do "import cv2" it'll still try to use the Python 2.7 version
provided in ROS, which will error. Thus, first try the version we just
installed:

    echo 'export PYTHONPATH="/usr/local/lib/python3.5/dist-packages:$PYTHONPATH"' >> ~/.bashrc

However, since that'll break Python 2, let's only do it for our one package (TODO?):

    echo "/usr/local/lib/python3.5/dist-packages/" >> TODO.pth

Make sure you have sourced the previous workspace before building:

    source /opt/ros/lunar/setup.bash
    source ~/catkin_ws/devel/setup.bash

Build everything:

    cd ~/catkin_py3
    workon tf-python3
    catkin_make -DFILTER=OFF -DPYTHON_EXECUTABLE=$(which python) -DPYTHON_VERSION=3
    catkin_make install

Now that this is an overlay workspace, you can source this:

    source ~/catkin_py3/devel/setup.bash
    echo "source ~/catkin_py3/devel/setup.bash" >> ~/.bashrc

## YOLO Setup
Copy the final weights over for YOLO into the *darknet_ros* directory:

    scp path/to/ras-object-detection/datasets/SmartHome/backup_100/SmartHome_final.weights \
        jetson:catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/SmartHome.weights
    scp path/to/ras-object-detection/datasets/SmartHome/config.cfg \
        jetson:catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/cfg/SmartHome.cfg
    scp path/to/ras-object-detection/dataset_100.data \
        jetson:catkin_ws/src/darknet_ros/darknet_ros/config/

Create *~/catkin_ws/src/darknet_ros/darknet_ros/config/SmartHome.yaml*,
changing the classes accordingly. Make sure the spaces/tabs are correct or else
it'll error parsing the file. Then change "yolo\_voc.yaml" to "SmartHome.yaml"
in *~/catkin_ws/src/darknet_ros/darknet_ros/launch/darknet_ros.launch*.

    yolo_model:
        config_file:
            name: SmartHome.cfg
        weight_file:
            name: SmartHome.weights
        threshold:
            value: 0.3
        detection_classes:
            names:
              -  food
              -  glass
              -  keys
              -  pillbottle
              -  plant
              -  umbrella
              -  watercan

If you don't want it showing the window with predictions, then set
*enable_opencv* and *use_darknet* to false in
*~/catkin_ws/src/darknet_ros/darknet_ros/config/ros.yaml*.

## Arduino Setup for Camera Pan/Tilt
Install Arduino IDE 1.0.6 on some computer (on Arch Linux try the *arduino10*
package in the AUR). Follow [ArbotiX-M instructions](http://learn.trossenrobotics.com/arbotix/7-arbotix-quick-start-guide).
Download the [ArbotiX-M](https://github.com/trossenrobotics/arbotix/archive/master.zip)
files and extract into your *~/sketchbook* folder.

If you can't get permissions to work despite adding yourself to uucp and lock
groups, then make sure that "/run/lock" is in the lock group:

    sudo chgrp lock /run/lock

Then, upload the File -> Sketchbook -> ArbotiX Sketches -> ros.

Setting up on the Jetson so you can control the servos from ROS:

    roslaunch ras_jetson camera.launch
    arbotix_gui

### Connecting to NUC
Since we'll run some on the Jetson and some on the NUC, we'll need to set one
up as the ROS master.  We'll use the NUC.

Jetson (since we enabled systemd-resolved earlier), so we can resolve the NUC
hostname with LLMNR:

    sudo apt install libnss-resolve

In *~/.bashrc* on the Jetson:

    export ROS_MASTER_URI=http://wsu-ras:11311

Then, replace 127.0.1.1 with 127.0.0.1 in */etc/hosts* on the Jetson.
Otherwise, often it can't connect to the ROS master on the NUC.

In *~/.bashrc* on the NUC:

    export ROS_MASTER_URI=http://wsu-ras:11311
    export TURTLEBOT_3D_SENSOR=astra
    export TURTLEBOT3_MODEL=waffle

Measure where the camera is relative to */base_link*, and then modify
*/opt/ros/kinetic/share/turtlebot_description/urdf/turtlebot_properties.urdf.xacro*
accordingly. For us on the TurtleBot 2, these are z = 1.4224 m and x = 0.0635 m.

Then, run on the NUC (TODO this was for TurtleBot 2):

    cd ~/catkin_ws/src
    git clone https://github.com/WSU-RAS/turtlebot3.git

    source ~/catkin_ws/devel/setup.bash
    roscore
    roslaunch turtlebot3_bringup turtlebot3_robot.launch
    roslaunch turtlebot3_bringup turtlebot3_remote.launch
    rosrun rviz rviz -d `rospack find turtlebot3_description`/rviz/model.rviz

Then, run on Jetson:

    roslaunch ras_jetson everything.launch

    # Either YOLO object detection
    roslaunch darknet_ros darknet_ros.launch

    # or TensorFlow object detection
    cd ~/catkin_py3
    . src/ras_jetson_py3/setup-env.sh
    roslaunch ras_jetson_py3 object_detector.launch

    # Optionally either of these, to control the camera:
    rosrun arbotix_python arbotix_gui
    rosrun object_detector_ros demo_pan_tilt.py

## Running Object Detection

### Speed

If you wish to set the clocks and GPU to full speed (and didn't enable this on
boot), then run
([src](https://devtalk.nvidia.com/default/topic/1018081/jetson-tx2/tensorflow-mobilenet-object-detection-model-in-tx2-is-very-slow-/post/5185487/),
[nvpmodel number reference](http://www.jetsonhacks.com/2017/03/25/nvpmodel-nvidia-jetson-tx2-development-kit/)):

    sudo nvpmodel -m 0
    sudo ~/jetson_clocks.sh

To check that they're running as fast as possible, check that @1300 is at the
end of the lines printed:

    sudo ~/jetson_clocks.sh --show

### Running YOLO

    roslaunch darknet_ros darknet_ros.launch

### Running TensorFlow

Run the Object Detector after editing the *params.yaml* file:

    cd ~/catkin_py3
    . src/ras_jetson_py3/setup-env.sh
    roslaunch ras_jetson_py3 object_detector.launch

Or, run components individually:

    roscore
    rosrun image_view image_view image:=/camera/rgb/image_raw
    rostopic echo /object_detector

    cd ~/catkin_py3
    . src/ras_jetson_py3/setup-env.sh
    rosrun ras_jetson_py3 object_detector.py \
        _graph_path:=~/networks/ssd_mobilenet_v1.pb \
        _labels_path:=~/networks/tf_label_map.pbtxt
