"""
OpenArm Kinematics using Placo library for forward and inverse kinematics.
Adapted for 7-DOF bimanual robot.
"""

import numpy as np
from pathlib import Path


class OpenArmKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        arm_prefix: str = "openarm_left_",
        target_frame_name: str = None,
    ):
        """
        Initialize placo-based kinematics solver for OpenArm.

        Args:
            urdf_path: Path to the robot URDF file
            arm_prefix: Prefix for arm joints ("openarm_left_" or "openarm_right_")
            target_frame_name: Name of the end-effector frame (default: {arm_prefix}hand_tcp)
        """
        try:
            import placo
        except ImportError as e:
            raise ImportError(
                "placo is required for OpenArmKinematics. "
                "Install with: pip install placo"
            ) from e

        self.arm_prefix = arm_prefix
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        # Default end-effector frame
        if target_frame_name is None:
            self.target_frame_name = f"{arm_prefix}hand_tcp"
        else:
            self.target_frame_name = target_frame_name

        # Joint names for this arm (7 DOF)
        self.joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 8)]

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(
            self.target_frame_name, np.eye(4)
        )

        # Joint limits from URDF (radians)
        self.joint_limits = self._get_joint_limits()

    def _get_joint_limits(self) -> dict:
        """Extract joint limits from the robot model."""
        limits = {}
        for joint_name in self.joint_names:
            # Default limits if not found
            limits[joint_name] = {
                "lower": -np.pi,
                "upper": np.pi,
            }
        return limits

    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration.

        Args:
            joint_pos_rad: Joint positions in radians (7 joints)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in radians (7 joints)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK

        Returns:
            Joint positions in radians that achieve the desired end-effector pose
        """
        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_pos[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task weights
        self.tip_frame.configure(
            self.target_frame_name, "soft", position_weight, orientation_weight
        )

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Extract joint positions
        joint_pos = np.array([
            self.robot.get_joint(joint_name)
            for joint_name in self.joint_names
        ])

        return joint_pos

    def reset(self, joint_pos_rad: np.ndarray = None):
        """
        Reset the internal state of the kinematics solver.

        Args:
            joint_pos_rad: Joint positions in radians to reset to.
                           If None, resets to zeros.
        """
        if joint_pos_rad is None:
            joint_pos_rad = np.zeros(len(self.joint_names))

        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        self.robot.update_kinematics()

    def get_jacobian(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix at the given joint configuration.

        Args:
            joint_pos_rad: Joint positions in radians (7 joints)

        Returns:
            6x7 Jacobian matrix (linear and angular velocities)
        """
        # Update joint positions
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        self.robot.update_kinematics()

        # Get full Jacobian (6 x N_dof) and extract only this arm's columns
        J_full = self.robot.frame_jacobian(self.target_frame_name, "world")
        cols = [self.robot.get_joint_v_offset(n) for n in self.joint_names]
        return J_full[:, cols]


class BimanualKinematics:
    """Wrapper for bimanual (two-arm) kinematics."""

    def __init__(self, urdf_path: str):
        """
        Initialize bimanual kinematics.

        Args:
            urdf_path: Path to the bimanual robot URDF file
        """
        self.left_arm = OpenArmKinematics(
            urdf_path=urdf_path,
            arm_prefix="openarm_left_",
        )
        self.right_arm = OpenArmKinematics(
            urdf_path=urdf_path,
            arm_prefix="openarm_right_",
        )

    def forward_kinematics(
        self, left_joints: np.ndarray, right_joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for both arms.

        Returns:
            Tuple of (left_ee_pose, right_ee_pose) as 4x4 matrices
        """
        left_pose = self.left_arm.forward_kinematics(left_joints)
        right_pose = self.right_arm.forward_kinematics(right_joints)
        return left_pose, right_pose

    def inverse_kinematics(
        self,
        current_left: np.ndarray,
        current_right: np.ndarray,
        desired_left_pose: np.ndarray,
        desired_right_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute inverse kinematics for both arms.

        Returns:
            Tuple of (left_joints, right_joints) in radians
        """
        left_joints = self.left_arm.inverse_kinematics(
            current_left, desired_left_pose, position_weight, orientation_weight
        )
        right_joints = self.right_arm.inverse_kinematics(
            current_right, desired_right_pose, position_weight, orientation_weight
        )
        return left_joints, right_joints
