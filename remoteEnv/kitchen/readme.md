
possible installations for gym and mujoco

sudo apt update && sudo apt install -y \
    libx11-dev libgl1-mesa-dev libglew-dev \
    libosmesa6-dev patchelf pkg-config
sudo apt-get update
sudo apt-get install libx11-dev
sudo apt-get install libglew-dev
sudo apt-get install patchelf
sudo apt-get install build-essential



after install the requirements 
upgrade gym==0.25.1

### python compatibility with mujoco_env
d4rl/kitchen/adept_envs/mujoco_env.py
collections.abc


### chain0.xml ###
Added only *eye_on_hand* camera on end effector following libero settings.
copy this to your D4RL path. (already done.)
desired place. (e.g., anaconda3/envs/kitchen_eval/lib/python3.8/site-packages/d4rl/kitchen/third_party/franka/assets/chain0.xml )

```xml
<body name="panda0_link7" pos="0.088 0 0" euler='1.57 0 0.7854'>
...
<body name='camera_holder' pos="0 0 0.1654" quat="0.92388 0 0 -0.382683" >
    <camera mode="fixed" name="eye_in_hand" pos="0.075 0 -0.10" quat="0 0.707108 0.707108 0" fovy="75"/>
</body>
...
</body>
```