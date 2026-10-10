# Test robots

Each folder keeps its upstream licence. Only collision geometry is included; visual meshes the
descriptions reference are not.

| Folder | Source | Licence | Changes |
|---|---|---|---|
| `franka/` | `franka_panda.urdf` from franka_ros 0.7.0; collision meshes (`meshes/collision/*.obj`) from NVlabs/curobo at `78fd485`, converted from franka_ros | Apache-2.0 (`franka/LICENSE`) | none |
| `franka/panda_collision.json` | Collision spheres, self-collision buffers and ignore pairs from NVlabs/curobo `franka.yml` | Apache-2.0, © NVIDIA | converted to batchplan's collision-model format |
| `ur5e/ur_description/` | UniversalRobots/Universal_Robots_ROS2_Description at `6662e15` (ur5e collision meshes, `package.xml`) | BSD-3-Clause (`ur5e/ur_description/LICENSE`) | `urdf/ur5e.urdf` expanded from `urdf/ur.urdf.xacro ur_type:=ur5e name:=ur5e` |
| `robotiq_2f85/robotiq_description/` | PickNikRobotics/ros2_robotiq_gripper at `a74d007` (2F-85 collision meshes, `package.xml`) | BSD-3-Clause (`robotiq_2f85/robotiq_description/LICENSE`) | `urdf/robotiq_2f_85.urdf` expanded from `urdf/robotiq_2f_85_gripper.urdf.xacro` |
| `so101/` | TheRobotStudio/SO-ARM100 at `a758567` (`Simulation/SO101/so101_new_calib.urdf`) | Apache-2.0 (`so101/LICENSE`) | Each mesh replaced by its convex hull (`assets/*_hull.stl`, about 0.15 MB instead of 15 MB): vertices snapped to a 2 mm grid, then every face pushed out until the hull contains all original vertices. The URDF points at the hulls. |

The UR5e and 2F-85 sit inside their ROS packages, so their `package://` mesh paths resolve by
searching upward for `package.xml`.
