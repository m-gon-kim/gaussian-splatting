
# from plyfile import PlyData, PlyElement
import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv
from scene import GaussianModel
from superpixel import SuperPixelManager
from arguments import PipelineParams
from gaussian_renderer import mg_render
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
import random
class GaussianMapper:
    def __init__(self, dataset):
        self.width = 640
        self.height = 480
        self.device = "cuda"
        with torch.no_grad():
            self.projection_matrix = None
            self.FoVx = None
            self.FoVy = None
            self.intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            self.inv_intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)

            # Wide angle parameters
            self.wide_ratio = 1.5
            self.projection_matrix_wide = None
            self.FoVx_wide = None
            self.FoVy_wide = None
            self.intr_wide = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            self.inv_intr_wide = torch.zeros((3, 3), dtype=torch.float32, device=self.device)

            self.recover = None
            self.uv_mid = None

        self.SetProjectionMatrix(dataset)
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))
        self.Flag_GS_Pause = False

        self.Flag_GS_Pause = False
        self.SetIntrinsics(dataset)

        self.SetSPMaskPoints()
        self.full_proj_transform_list = []
        self.world_view_transform_list = []
        self.camera_center_list = []

        # from images
        self.SP_rgb_list = []
        self.SP_img_gt_list = []
        self.SP_xyz_list = []  # Converted from Depth map
        self.SP_superpixel_list = []
        self.iteration_frame = []

        with torch.no_grad():
            self.SP_poses = torch.empty((4, 4, 0), dtype=torch.float32, device=self.device)

        # points (2D, 3D)
        self.SP_ref_3d_list = []
        self.SP_ref_color_list = []
        self.SP_global_3d_list = []

        # Super pixel
        self.sp_manager = SuperPixelManager(self.width, self.height)
        # Gaussians
        self.gaussian = GaussianModel(3, self.device)
        with torch.no_grad():
            self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
            #Mask
        self.cam_centers = []
        self.cameras_extent = 0
        self.size_threshold = 20
        self.densify_grad_threshold = 0.0002
        self.loss_threshold = 0.1



        self.iteration = 1
        self.densification_interval = 1
        # viz
        self.viz_full_proj_transform_list = []
        self.viz_world_view_transform_list = []
        self.viz_camera_center_list = []
        self.fixed_world_view_transform_list = []
        self.third_full_proj_transform_list = []
        self.third_world_view_transform = None
        self.third_camera_center = None
        self.third_projected_camera_center = None
        self.wireframe_camera_positions = []
        self.wireframe_camera_index = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=torch.int64, device=self.device)
        self.SetVizParams()
        self.loss_dict = {}




    def SetIntrinsics(self, dataset):
        fx, fy, cx, cy = dataset.get_camera_intrinsic()
        # fx = 535.4
        # fy = 539.2
        # cx = 320.1
        # cy = 247.6

        self.intr[0][0] = fx
        self.intr[0][2] = cx
        self.intr[1][1] = fy
        self.intr[1][2] = cy
        self.intr[2][2] = 1

        self.inv_intr[0][0] = 1 / fx
        self.inv_intr[0][2] = -cx / fx
        self.inv_intr[1][1] = 1 / fy
        self.inv_intr[1][2] = -cy / fy
        self.inv_intr[2][2] = 1

        self.intr_wide[0][0] = fx / self.wide_ratio
        self.intr_wide[0][2] = cx
        self.intr_wide[1][1] = fy / self.wide_ratio
        self.intr_wide[1][2] = cy
        self.intr_wide[2][2] = 1

        self.inv_intr_wide[0][0] = 1 / (fx / self.wide_ratio)
        self.inv_intr_wide[0][2] = -cx / (fx / self.wide_ratio)
        self.inv_intr_wide[1][1] = 1 / (fy / self.wide_ratio)
        self.inv_intr_wide[1][2] = -cy / (fy / self.wide_ratio)
        self.inv_intr_wide[2][2] = 1

    def SetSPMaskPoints(self):
        with torch.no_grad():
            mid_xyz = torch.zeros((3, 1), dtype=torch.float32, device=self.device)
            mid_xyz[2] = 5
            uv_mid = torch.matmul(self.intr, mid_xyz)
            self.uv_mid = uv_mid/uv_mid[2, :]
            self.recover = torch.zeros((4, 1),dtype=torch.float32, device=self.device)
            self.recover[2] = 10.0
            self.recover[3] = 1.0

    def SetProjectionMatrix(self, dataset):
        fx, fy = dataset.get_camera_intrinsic()[:2]

        FoVx = 2 * math.atan(640 / (2 * fx))
        FoVy = 2 * math.atan(480 / (2 * fy))
        FoVx_wide = 2 * math.atan(640 / (2 * fx / self.wide_ratio))
        FoVy_wide = 2 * math.atan(480 / (2 * fy / self.wide_ratio))
        with torch.no_grad():
            self.FoVx = torch.tensor(FoVx, dtype=torch.float32, device=self.device)
            self.FoVy = torch.tensor(FoVy, dtype=torch.float32, device=self.device)
            self.projection_matrix = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).type(torch.FloatTensor).to(self.device)

            self.FoVx_wide = torch.tensor(FoVx_wide, dtype=torch.float32, device=self.device)
            self.FoVy_wide = torch.tensor(FoVy_wide, dtype=torch.float32, device=self.device)
            self.projection_matrix_wide = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx_wide, fovY=FoVy_wide).transpose(0, 1).type(torch.FloatTensor).to(self.device)

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        with torch.no_grad():
            P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def getNerfppNorm(self, pose):
        def get_center_and_diag():
            cam_centers = np.hstack(self.cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_center = pose[:3, 3:4].detach().cpu().numpy()
        self.cam_centers.append(cam_center)

        center, diagonal = get_center_and_diag()
        radius = diagonal * 1.1

        translate = -center
        self.cameras_extent = radius
        print("Radius", radius)
        return {"translate": translate, "radius": radius}

    def TMPConvertCamera2World(self, xyz, pose):
        with torch.no_grad():
            ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=self.device)
            xyz_one = torch.cat((xyz, ones), dim=0)
            world_xyz = torch.matmul(pose, xyz_one)

            xyz_mask = world_xyz[3, :].ne(0.0)
            masked_world_xyz = world_xyz[:, xyz_mask]

            masked_world_xyz = masked_world_xyz[:, :] / masked_world_xyz[3, :]

        return masked_world_xyz

    def SetVizParams(self):
        with torch.no_grad():
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            camera_center = pose[3, :3].detach()
            world_view_transform = torch.inverse(pose).detach()
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)
        # self.viz_full_proj_transform_list.append((world_view_transform.detach().cpu().unsqueeze(0).bmm(
        #     (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
        #     self.device))
        self.viz_full_proj_transform_list.append(full_proj_transform.detach())
        self.viz_world_view_transform_list.append(world_view_transform.detach())
        self.viz_camera_center_list.append(camera_center.detach())
        # self.fixed_world_view_transform_list

        #
        # with torch.no_grad():
        #     pose2 = torch.eye(4, dtype=torch.float32, device=self.device)
        #     pose2[2, 3] = 4.8
        #     pose2[1, 3] = -3.6
        #
        #     pose2[0, 0] = -1
        #     pose2[1, 1] = 0.5
        #     pose2[1, 2] = 0.866
        #     pose2[2, 1] = 0.866
        #     pose2[2, 2] = -0.5
        #
        #
        #     camera_center2 = pose2.T[3, :3].detach()
        #     world_view_transform2 = torch.inverse(pose2).T.detach()
        #     full_proj_transform2 = torch.matmul(world_view_transform2, self.projection_matrix)
        #     # self.viz_full_proj_transform_list.append((world_view_transform2.detach().cpu().unsqueeze(0).bmm(
        #     #     (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
        #     #     self.device))
        # self.viz_full_proj_transform_list.append(full_proj_transform2.detach())
        # self.viz_world_view_transform_list.append(world_view_transform2.detach())
        # self.viz_camera_center_list.append(camera_center2.detach())
        #
        # with torch.no_grad():
        #     pose = torch.eye(4, dtype=torch.float32, device=self.device)
        #     pose[2, 3] = -5
        #     pose[1, 3] = -0.5
        #     camera_center = pose.T[3, :3].detach()
        #     world_view_transform = torch.inverse(pose).T.detach()
        # self.viz_full_proj_transform_list.append((world_view_transform.detach().cpu().unsqueeze(0).bmm(
        #     (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
        #     self.device))
        # self.viz_world_view_transform_list.append(world_view_transform.detach())
        # self.viz_camera_center_list.append(camera_center.detach())

        with torch.no_grad():
            pose2 = torch.eye(4, dtype=torch.float32, device=self.device)
            x_rot_degree = -30
            y_rot_degree = 0
            z_rot_degree = 0
            x_rot = x_rot_degree * (math.pi / 180)
            y_rot = y_rot_degree * (math.pi / 180)
            z_rot = z_rot_degree * (math.pi / 180)

            x_rot_mat = torch.tensor([[1, 0, 0], [0, math.cos(x_rot), -math.sin(x_rot)], [0, math.sin(x_rot), math.cos(x_rot)]], dtype=torch.float32, device=self.device)
            y_rot_mat = torch.tensor([[math.cos(y_rot), 0, math.sin(y_rot)], [0, 1, 0], [-math.sin(y_rot), 0, math.cos(y_rot)]], dtype=torch.float32, device=self.device)
            z_rot_mat = torch.tensor([[math.cos(z_rot), -math.sin(z_rot), 0], [math.sin(z_rot), math.cos(z_rot), 0], [0, 0, 1]], dtype=torch.float32, device=self.device)
            rot = torch.matmul(torch.matmul(x_rot_mat, y_rot_mat), z_rot_mat)
            pose2[:3, :3] = rot

            trans = torch.tensor([0.3, -1.5, -2.5], dtype=torch.float32, device=self.device)
            pose2[:3, 3] = trans

            camera_center2 = pose2.T[3, :3].detach()
            world_view_transform2 = torch.inverse(pose2).T.detach()
            full_proj_transform2 = torch.matmul(world_view_transform2, self.projection_matrix)
            # self.viz_full_proj_transform_list.append((world_view_transform2.detach().cpu().unsqueeze(0).bmm(
            #     (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            #     self.device))
        self.viz_full_proj_transform_list.append(full_proj_transform2.detach())
        self.viz_world_view_transform_list.append(world_view_transform2.detach())
        self.viz_camera_center_list.append(camera_center2.detach())

    def SetThirdPersonViewCamera(self, pose):
        with torch.no_grad():
            rel_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            tvec = torch.tensor([0, 0, -0.5], dtype=torch.float32, device=self.device)
            rot = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=self.device)
            rel_pose[:3, :3] = rot
            rel_pose[:3, 3] = tvec
            rel_pose = torch.matmul(pose, rel_pose)
            camera_center = rel_pose.T[3, :3].detach()
            world_view_transform = torch.inverse(rel_pose).T.detach()
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix_wide)
        self.third_world_view_transform = rel_pose.detach()
        self.third_full_proj_transform = full_proj_transform.detach()
        self.third_camera_center = camera_center.detach()

    def ProjectThirdPersonViewCameraPositions(self):
        # Draw wireframe camera with opencv
        projected_camera_centers = []
        with torch.no_grad():
            for i in range(len(self.camera_center_list)):
                cam_center = self.camera_center_list[i]
                cam_center_4d = torch.cat((cam_center, torch.tensor([1], dtype=torch.float32, device=self.device)))

                view_space_pos = torch.matmul(torch.inverse(self.third_world_view_transform), cam_center_4d)
                ndc_space_pos = view_space_pos / view_space_pos[3]
                cam_uv = torch.matmul(self.intr, ndc_space_pos[:3])
                cam_uv = cam_uv / cam_uv[2]
                projected_camera_centers.append(cam_uv[:2])
        return projected_camera_centers

    def CreateCameraWireframePoints(self):
        with torch.no_grad():
            pose = self.SP_poses[:, :, -1]

            c_depth = 0.0001
            c_width = self.width * c_depth
            c_height = self.height * c_depth

            tvec1 = torch.tensor([c_width, c_height, c_depth*400], dtype=torch.float32, device=self.device)
            tvec2 = torch.tensor([-c_width, c_height, c_depth*400], dtype=torch.float32, device=self.device)
            tvec3 = torch.tensor([-c_width, -c_height, c_depth*400], dtype=torch.float32, device=self.device)
            tvec4 = torch.tensor([c_width, -c_height, c_depth*400], dtype=torch.float32, device=self.device)

            rel_pose1 = torch.eye(4, dtype=torch.float32, device=self.device)
            rel_pose2 = torch.eye(4, dtype=torch.float32, device=self.device)
            rel_pose3 = torch.eye(4, dtype=torch.float32, device=self.device)
            rel_pose4 = torch.eye(4, dtype=torch.float32, device=self.device)
            rel_pose1[:3, 3] = tvec1
            rel_pose2[:3, 3] = tvec2
            rel_pose3[:3, 3] = tvec3
            rel_pose4[:3, 3] = tvec4

            rel_pose1 = torch.matmul(pose, rel_pose1)
            rel_pose2 = torch.matmul(pose, rel_pose2)
            rel_pose3 = torch.matmul(pose, rel_pose3)
            rel_pose4 = torch.matmul(pose, rel_pose4)

            pose1 = rel_pose1.T[3, :3].detach()
            pose2 = rel_pose2.T[3, :3].detach()
            pose3 = rel_pose3.T[3, :3].detach()
            pose4 = rel_pose4.T[3, :3].detach()

            poses = [pose.T[3, :3].detach(), pose1, pose2, pose3, pose4]

            self.wireframe_camera_positions.append(poses)

    def DrawCameraWireframes(self, img, tvec, intr):
        with torch.no_grad():
            for i in range(len(self.wireframe_camera_positions)):
                poses = self.wireframe_camera_positions[i]
                points = []
                # Calculate 2D points
                for p in poses:
                    pose_4d = torch.eye(4, dtype=torch.float32, device=self.device)
                    pose_4d[:3, 3] = tvec

                    p_4d = torch.cat((p, torch.tensor([1], dtype=torch.float32, device=self.device)))
                    view_space_pos = torch.matmul(torch.inverse(pose_4d), p_4d)
                    ndc_space_pos = view_space_pos / view_space_pos[3]

                    cam_uv = torch.matmul(intr, ndc_space_pos[:3])
                    cam_uv = cam_uv / cam_uv[2]
                    points.append(cam_uv[:2])

                # Draw wireframe camera
                for j in range(len(self.wireframe_camera_index)):
                    point1 = points[self.wireframe_camera_index[j][0]]
                    point2 = points[self.wireframe_camera_index[j][1]]
                    cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 1)

                # Draw line between current and previous camera
                if i > 0:
                    l_p1 = points[0]
                    w_l_p2 = self.wireframe_camera_positions[i-1][0]
                    w_l_p2_4d = torch.cat((w_l_p2, torch.tensor([1], dtype=torch.float32, device=self.device)))
                    view_space_l_p2 = torch.matmul(torch.inverse(pose_4d), w_l_p2_4d)
                    ndc_space_l_p2 = view_space_l_p2 / view_space_l_p2[3]
                    l_p2 = torch.matmul(intr, ndc_space_l_p2[:3])
                    l_p2 = l_p2 / l_p2[2]
                    l_p2 = l_p2[:2]
                    cv2.line(img, (int(l_p1[0]), int(l_p1[1])), (int(l_p2[0]), int(l_p2[1])), (0, 0, 255), 1)

        return img

    def CreateInitialKeyframe(self, rgb, SP_xyz, pose):
        with torch.no_grad():
            rgb_torch = torch.from_numpy(rgb).to(self.device)
            img_gt = torch.permute(rgb_torch.type(torch.FloatTensor), (2, 0, 1)).to(self.device) / 255.0

            self.SP_poses = torch.cat((self.SP_poses, pose.unsqueeze(dim=2)), dim=2)
            super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)

            mask = super_pixel_index[0, :].gt(0)
            masked_index = super_pixel_index[:, mask]
            mask = masked_index[1, :].gt(0)
            masked_index = masked_index[:, mask]
            mask = masked_index[0, :].lt(self.height-1)
            masked_index = masked_index[:, mask]
            mask = masked_index[1, :].lt(self.width-1)
            masked_index = masked_index[:, mask]



            masked_xyz = SP_xyz[masked_index[0, :], masked_index[1, :], :].T
            masked_rgb = rgb_torch[masked_index[0, :], masked_index[1, :], :]

            mask = masked_xyz[2, :].gt(0.2)
            masked_xyz = masked_xyz[:, mask]
            masked_rgb = masked_rgb[mask, :]

            point_list_for_gaussian = self.TMPConvertCamera2World(masked_xyz, pose)
            del masked_index
            del masked_xyz

            world_view_transform = torch.inverse(pose).T.detach()
            camera_center = torch.inverse(world_view_transform)[3, :3]
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)

        self.gaussian.AddGaussian(point_list_for_gaussian[:3, :], masked_rgb, len(self.SP_img_gt_list))
        del masked_rgb
        self.SP_global_3d_list.append(point_list_for_gaussian[:3, :])
        self.SP_img_gt_list.append(img_gt)

        self.full_proj_transform_list.append(full_proj_transform.detach())
        self.world_view_transform_list.append(world_view_transform.detach())
        self.camera_center_list.append(camera_center.detach())
        self.iteration_frame.append(0)
        del full_proj_transform
        del world_view_transform
        del camera_center

        # Gaussian
        self.gaussian.InitializeOptimizer()
        self.loss_dict[0] = 1.0
        self.CreateCameraWireframePoints()

    def CreateKeyframe(self, rgb, SP_xyz, pose):
        print("KFrame")
        w_padding = 100
        h_padding = 50

        min_u = 0
        min_v = 0
        max_u = 0
        max_v = 0

        with torch.no_grad():
            prev_pose = self.SP_poses[:, :, -1]
            tvec = pose[:3, 3] - prev_pose[:3, 3]
            rot =  torch.matmul(pose[:3, :3], (prev_pose[:3, :3]).T)
            relativepose = torch.eye((4), dtype=torch.float32, device=self.device)
            relativepose[:3, :3] = rot
            relativepose[:3, 3] = tvec
            relative_points = torch.matmul(relativepose, self.recover.detach())[:3, :]
            uv = torch.matmul(self.intr, relative_points)
            uv = uv[:, :] / uv[2, :]
            diff = uv - self.uv_mid
            if diff[0] > 0:  # 카메라가 왼쪽으로 움직였음
                min_u = 0
                max_u = w_padding + diff[0]
            else:  # 카메라가 오른쪽으로 움직였음
                min_u = self.width -1 - w_padding + diff[0]
                max_u = self.width - 1
            if diff[1] > 0:  # 카메라가 위로 움직였음
                min_v = 0
                max_v = h_padding + diff[1]
            else:  # 카메라가 아래로 움직였음
                min_v = self.height - 1 - h_padding + diff[1]
                max_v = self.height - 1

            # 이전 프레임에서의 pointcloud를 현재 프레임에 projection하고, BB를 만들었음
            rgb_torch = torch.from_numpy(rgb).detach().to(self.device)
            img_gt = torch.permute(rgb_torch.type(torch.FloatTensor), (2, 0, 1)).to(self.device)/255.0

            super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
            mask_min_u = super_pixel_index[1, :].ge(min_u)
            mask_max_u = super_pixel_index[1, :].le(max_u)
            mask_min_v = super_pixel_index[0, :].ge(min_v)
            mask_max_v = super_pixel_index[0, :].le(max_v)
            mask = torch.logical_and(torch.logical_and(mask_min_u, mask_max_u),
                                    torch.logical_and(mask_min_v, mask_max_v))
            masked_index = super_pixel_index[:, ~mask]

            masked_xyz = SP_xyz[masked_index[0, :], masked_index[1, :], :].T
            masked_rgb = rgb_torch[masked_index[0, :], masked_index[1, :], :]

            mask = masked_xyz[2, :].gt(0.2)
            masked_xyz = masked_xyz[:, mask]
            masked_rgb = masked_rgb[mask, :]



        if masked_xyz.shape[1] > 0:
            with torch.no_grad():
                point_list_for_gaussian = self.TMPConvertCamera2World(masked_xyz, (pose))

            self.gaussian.AddGaussian(point_list_for_gaussian[:3, :], masked_rgb, len(self.SP_img_gt_list)+1)
            self.SP_global_3d_list.append(point_list_for_gaussian[:3, :])
        with torch.no_grad():
            world_view_transform = torch.inverse(pose).T.detach()
            camera_center = torch.inverse(world_view_transform)[3, :3]
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)
            self.SP_poses = torch.cat((self.SP_poses, pose.unsqueeze(dim=2)), dim=2)

        self.SP_img_gt_list.append(img_gt)
        self.full_proj_transform_list.append(full_proj_transform.detach())
        self.world_view_transform_list.append(world_view_transform.detach())
        self.camera_center_list.append(camera_center.detach())
        self.iteration_frame.append(0)

        # Gaussian
        self.gaussian.InitializeOptimizer()
        self.loss_dict[self.SP_poses.shape[2]-1] = 1.0
        self.CreateCameraWireframePoints()



    def InsertionOptimize(self):
        lambda_dssim = 0.2
        optimization_i_threshold = 10
        index = len(self.SP_img_gt_list)-1
        # if index > 119 and index < 180:
        #     return
        for optimization_i in range(optimization_i_threshold):
            cntr = 0
            img_gt = self.SP_img_gt_list[index].detach()
            with torch.no_grad():
                world_view_transform = self.world_view_transform_list[index]
                full_proj_transform = self.full_proj_transform_list[index]
                camera_center = self.camera_center_list[index]
            render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, world_view_transform,
                                   full_proj_transform, camera_center, self.gaussian, self.pipe, self.background,
                                   1.0)
            img, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            Ll1 = l1_loss(img, img_gt)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
            self.loss_dict[index] = float(loss.detach())

            loss.backward()

            self.gaussian.max_radii2D[visibility_filter] = torch.max(self.gaussian.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
            self.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if index%2 == 0 and index > 0 and optimization_i == optimization_i_threshold-1 :
                print(f"PRUNE {self.iteration} {self.densification_interval}")
                self.densification_interval = 0
                self.gaussian.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent,
                                                self.size_threshold)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            cntr+=1
            del img

    def FullOptimizeGaussian(self, Flag_densification):
        if self.SP_poses.shape[2] == 0:
            return
        lambda_dssim = 0.2
        sample_kf_index_list = []

        sample_kf_index_list = list(range(self.SP_poses.shape[2]))

        # self.gaussian.update_learning_rate(self.iteration)
        optimization_i_threshold = 10
        for optimization_i in range(optimization_i_threshold):
            for i in sample_kf_index_list:
                img_gt = self.SP_img_gt_list[i].detach()
                with torch.no_grad():
                    world_view_transform = self.world_view_transform_list[i]
                    full_proj_transform = self.full_proj_transform_list[i]
                    camera_center = self.camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, world_view_transform,
                                       full_proj_transform, camera_center, self.gaussian, self.pipe, self.background,
                                       1.0)
                img, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                Ll1 = l1_loss(img, img_gt)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
                self.loss_dict[i] = float(loss.detach())

                loss.backward()

                self.gaussian.max_radii2D[visibility_filter] = torch.max(self.gaussian.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if Flag_densification and i%4 == 0 and optimization_i == int(optimization_i_threshold/2):
                    self.gaussian.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent,
                                                    self.size_threshold)

                self.gaussian.optimizer.step()
                self.gaussian.optimizer.zero_grad(set_to_none=True)
                del img
        torch.cuda.empty_cache()
    def OptimizeGaussian(self, Flag_densification):
        if self.Flag_GS_Pause:
            return
        if self.SP_poses.shape[2] == 0:
            return
        lambda_dssim = 0.2
        kf_cnt_threshold=3
        kf_cnt_sample=3

        self.iteration+=1
        if self.SP_poses.shape[2] <= kf_cnt_threshold + kf_cnt_sample:
            sample_kf_index_list = list(range(self.SP_poses.shape[2]))
        else:
            kf_sorted_by_loss = list(dict(sorted(self.loss_dict.items(), key=lambda x: x[1], reverse=True)))
            sample_kf_index_list = kf_sorted_by_loss[:3]

            sample_kf_index = random.sample(range(kf_cnt_threshold, self.SP_poses.shape[2]),
                                            kf_cnt_sample)
            sample_kf_index_list += [kf_sorted_by_loss[i] for i in sample_kf_index]
            # print("sample_kf_index_list", sample_kf_index_list)

        # self.gaussian.update_learning_rate(self.iteration)
        optimization_i_threshold = 10
        for optimization_i in range(optimization_i_threshold):
            for i in sample_kf_index_list:
                # if i > 119 and i < 180:
                #     continue
                img_gt = self.SP_img_gt_list[i].detach()
                with torch.no_grad():
                    world_view_transform = self.world_view_transform_list[i]
                    full_proj_transform = self.full_proj_transform_list[i]
                    camera_center = self.camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, world_view_transform,
                                       full_proj_transform, camera_center, self.gaussian, self.pipe, self.background,
                                       1.0)
                img, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                Ll1 = l1_loss(img, img_gt)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
                self.loss_dict[i] = float(loss.detach())

                loss.backward()

                self.gaussian.max_radii2D[visibility_filter] = torch.max(self.gaussian.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                self.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if Flag_densification and i%10 == 0 and optimization_i == (0):
                    print(f"PRUNE {self.iteration} ")
                    self.gaussian.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent,
                                                    self.size_threshold)

                self.gaussian.optimizer.step()
                self.gaussian.optimizer.zero_grad(set_to_none=True)
                del img
        torch.cuda.empty_cache()



    def Visualize(self):
        # print("Viz len: ", self.SP_poses.shape[2])
        if self.SP_poses.shape[2] > 0:

            # Fixed camera position for visualization
            for i in range(len(self.viz_world_view_transform_list)):
                viz_world_view_transform = self.viz_world_view_transform_list[i]
                viz_full_proj_transform = self.viz_full_proj_transform_list[i]
                viz_camera_center = self.viz_camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, viz_world_view_transform, viz_full_proj_transform,
                                       viz_camera_center, self.gaussian, self.pipe, self.background, 1.0)
                img = render_pkg["render"]  #GRB
                np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy().copy()    #RGB
                if i == 1:
                    self.DrawCameraWireframes(np_render, viz_camera_center, self.intr)
                cv2.imshow(f"start_gs{i}", np_render)

            # Render from keyframes
            for i in range(0, self.SP_poses.shape[2], 3):
                world_view_transform = self.world_view_transform_list[i]
                full_proj_transform = self.full_proj_transform_list[i]
                camera_center = self.camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, world_view_transform,
                                       full_proj_transform,
                                       camera_center, self.gaussian, self.pipe, self.background, 1.0)
                img = render_pkg["render"]
                np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()
                cv2.imshow(f"rendered{int(i*10 / 3)}", np_render)

            # Render all frames with predicted camera poses
            frame = self.SP_poses.shape[2]-1
            all_world_view_transform = self.world_view_transform_list[frame]
            all_full_proj_transform = self.full_proj_transform_list[frame]
            all_camera_center = self.camera_center_list[frame]

            self.SetThirdPersonViewCamera(self.SP_poses[:, :, frame])
            third_world_view_transform = self.third_world_view_transform
            third_full_proj_transform = self.third_full_proj_transform
            third_w_center = self.third_camera_center

            # projected_camera_centers = self.ProjectThirdPersonViewCameraPositions()

            render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, all_world_view_transform, all_full_proj_transform,
                                   all_camera_center, self.gaussian, self.pipe, self.background, 1.0)
            img = render_pkg["render"]
            # print(img)
            np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()
            cv2.imshow(f"sw", np_render)

            render_third_pkg = mg_render(self.FoVx_wide, self.FoVy_wide, self.height, self.width, third_world_view_transform, third_full_proj_transform,
                                      third_w_center, self.gaussian, self.pipe, self.background, 1.0)
            img_third = render_third_pkg["render"]
            # print(img)
            np_render_third = torch.permute(img_third, (1, 2, 0)).detach().cpu().numpy().copy()

            # Draw wireframe camera positions with opencv
            # for point in projected_camera_centers:
            #     # img_center = np.array([self.width // 2, self.height // 2])
            #     projected_camera_center = (int(point[0]), int(point[1]))
            #     cv2.circle(np_render_third, projected_camera_center, 2, (0, 0, 255), -1)

            self.DrawCameraWireframes(np_render_third, third_w_center, self.intr)

            cv2.imshow(f"third", np_render_third)

            cv2.waitKey(1)

    def GaussianMap(self, mapping_result_instance):
        if not mapping_result_instance[0]:
            return
        mapping_result = mapping_result_instance[1]
        status = mapping_result[0]

        self.Flag_GS_Pause = status[4]

        if (not status[0]) and (not status[1]) and (not status[2]):
            return

        if status[0] or status[1]:
            sensor = mapping_result[1]
            rgb = sensor[0]
            SP_xyz = sensor[1]
            with torch.no_grad():
                xyz_t = torch.from_numpy(SP_xyz).to(self.device)
                pose = mapping_result[2].to(self.device)  # torch.tensor

            if status[0] and status[1] :  # First KF
                self.CreateInitialKeyframe(rgb, xyz_t, pose)  # rgb must be numpy (Super pixel)
                self.getNerfppNorm(pose)
            elif status[0] and not (status[1]):  # Not First Frame
                self.CreateKeyframe(rgb, xyz_t, pose)
                self.getNerfppNorm(pose)

            if status[2]: # BA
                with torch.no_grad():
                    BA_results = mapping_result[3]
                    SP_poses = BA_results.detach().to(self.device)  # torch
                    self.SP_poses[:, :, :SP_poses.shape[2]] = SP_poses

                    for i in range(SP_poses.shape[2]):
                        pose = SP_poses[:, :, i]
                        # pre-compute poses
                        world_view_transform = torch.inverse(pose).T
                        camera_center = torch.inverse(world_view_transform)[3, :3]
                        full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)

                        self.full_proj_transform_list[i] = full_proj_transform.detach()
                        self.world_view_transform_list[i] = world_view_transform.detach()
                        self.camera_center_list[i] = camera_center.detach()

                self.FullOptimizeGaussian(status[3])


        elif status[2]:  # BA
            with torch.no_grad():
                BA_results = mapping_result[3]
                SP_poses = BA_results.detach().to(self.device)  # torch
                self.SP_poses[:, :, :SP_poses.shape[2]] = SP_poses

                for i in range(SP_poses.shape[2]):
                    pose = SP_poses[:, :, i]
                    # pre-compute poses
                    world_view_transform = torch.inverse(pose).T
                    camera_center = torch.inverse(world_view_transform)[3, :3]
                    full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)

                    self.full_proj_transform_list[i] = full_proj_transform.detach()
                    self.world_view_transform_list[i] = world_view_transform.detach()
                    self.camera_center_list[i] = camera_center.detach()

            self.FullOptimizeGaussian(status[3])


        return


