
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
    def __init__(self):
        self.width = 640
        self.height = 480
        self.device = "cuda"
        with torch.no_grad():
            self.projection_matrix = None
            self.FoVx = None
            self.FoVy = None
            self.intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            self.inv_intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            self.recover = None
            self.uv_mid = None
        self.SetProjectionMatrix()
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))
        self.Flag_GS_Pause = False

        self.SetIntrinsics()
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
        self.SetVizParams()
        self.loss_dict = {}




    def SetIntrinsics(self):
        fx = 535.4
        fy = 539.2
        cx = 320.1
        cy = 247.6

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

    def SetSPMaskPoints(self):
        with torch.no_grad():
            mid_xyz = torch.zeros((3, 1), dtype=torch.float32, device=self.device)
            mid_xyz[2] = 5
            uv_mid = torch.matmul(self.intr, mid_xyz)
            self.uv_mid = uv_mid/uv_mid[2, :]
            self.recover = torch.zeros((4, 1),dtype=torch.float32, device=self.device)
            self.recover[2] = 10.0
            self.recover[3] = 1.0

    def SetProjectionMatrix(self):
        fx = 535.4
        fy = 539.2

        FoVx = 2 * math.atan(640 / (2 * fx))
        FoVy = 2 * math.atan(480 / (2 * fy))
        with torch.no_grad():
            self.FoVx = torch.tensor(FoVx, dtype=torch.float32, device=self.device)
            self.FoVy = torch.tensor(FoVy, dtype=torch.float32, device=self.device)

            self.projection_matrix = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).type(torch.FloatTensor).to(self.device)

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


        with torch.no_grad():
            pose2 = torch.eye(4, dtype=torch.float32, device=self.device)
            pose2[2, 3] = 4.8
            pose2[1, 3] = -3.6

            pose2[0, 0] = -1
            pose2[1, 1] = 0.5
            pose2[1, 2] = 0.866
            pose2[2, 1] = 0.866
            pose2[2, 2] = -0.5


            camera_center2 = pose2.T[3, :3].detach()
            world_view_transform2 = torch.inverse(pose2).T.detach()
            full_proj_transform2 = torch.matmul(world_view_transform2, self.projection_matrix)
            # self.viz_full_proj_transform_list.append((world_view_transform2.detach().cpu().unsqueeze(0).bmm(
            #     (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            #     self.device))
        self.viz_full_proj_transform_list.append(full_proj_transform2.detach())
        self.viz_world_view_transform_list.append(world_view_transform2.detach())
        self.viz_camera_center_list.append(camera_center2.detach())

        with torch.no_grad():
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            pose[2, 3] = -5
            pose[1, 3] = -0.5
            camera_center = pose.T[3, :3].detach()
            world_view_transform = torch.inverse(pose).T.detach()
        self.viz_full_proj_transform_list.append((world_view_transform.detach().cpu().unsqueeze(0).bmm(
            (self.projection_matrix).detach().cpu().unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device))
        self.viz_world_view_transform_list.append(world_view_transform.detach())
        self.viz_camera_center_list.append(camera_center.detach())

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


        # self.SP_ref_3d_list.append(sp_3d_list)
        # self.SP_ref_color_list.append(sp_color_list)

        # pre-compute poses
            world_view_transform = torch.inverse(pose).T.detach()
            camera_center = torch.inverse(world_view_transform)[3, :3]
        # print(f"bmm: {world_view_transform.unsqueeze(0).shape}, {self.projection_matrix.detach().cpu().unsqueeze(0).shape}")
        # full_proj_transform = world_view_transform.unsqueeze(0).cpu().bmm((self.projection_matrix).detach().cpu().unsqueeze(0))
        # print(f"full_proj_transform {full_proj_transform.shape}")
        # full_proj_transform = full_proj_transform.squeeze(0).to(self.device)
        # print(f"full_proj_transform squ {full_proj_transform.shape}")
            full_proj_transform = torch.matmul(world_view_transform, self.projection_matrix)
        # world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        # camera_center = camera_center.type(torch.FloatTensor).to(self.device)

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
            mask = torch.logical_or(torch.logical_and(mask_min_u, mask_max_u),
                                    torch.logical_and(mask_min_v, mask_max_v))
            masked_index = super_pixel_index[:, mask]
            #
            # mask = super_pixel_index[0, :].ge(0)
            # masked_index = super_pixel_index[:, mask]
            # mask = masked_index[1, :].ge(0)
            # masked_index = masked_index[:, mask]
            # mask = masked_index[0, :].le(self.height-1)
            # masked_index = masked_index[:, mask]
            # mask = masked_index[1, :].le(self.width-1)
            # masked_index = masked_index[:, mask]
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
            if index%10 == 0 and index > 0 and optimization_i == optimization_i_threshold-1 :
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

        self.iteration+=1
        print("OPTIMIZE")
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
                if Flag_densification and i%30 == 0 and optimization_i == (optimization_i_threshold-1):
                    print(f"PRUNE {self.iteration} ")
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
        sample_kf_index_list = []

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
                if Flag_densification and i%20 == 0 and optimization_i == (optimization_i_threshold-1):
                    print(f"PRUNE {self.iteration} ")
                    self.gaussian.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent,
                                                    self.size_threshold)

                self.gaussian.optimizer.step()
                self.gaussian.optimizer.zero_grad(set_to_none=True)
                del img
        torch.cuda.empty_cache()



    def Visualize(self):
        if self.SP_poses.shape[2] > 0:
            for i in range(len(self.viz_world_view_transform_list)):
                viz_world_view_transform = self.viz_world_view_transform_list[i]
                viz_full_proj_transform = self.viz_full_proj_transform_list[i]
                viz_camera_center = self.viz_camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, viz_world_view_transform, viz_full_proj_transform,
                                       viz_camera_center, self.gaussian, self.pipe, self.background, 1.0)
                img = render_pkg["render"]
            # print(img)
                np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()
                cv2.imshow(f"start_gs{i}", np_render)
            for i in range(0, self.SP_poses.shape[2], 10):
                viz_world_view_transform = self.world_view_transform_list[i]
                viz_full_proj_transform = self.full_proj_transform_list[i]
                viz_camera_center = self.camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, self.height, self.width, viz_world_view_transform,
                                       viz_full_proj_transform,
                                       viz_camera_center, self.gaussian, self.pipe, self.background, 1.0)
                img = render_pkg["render"]
                # print(img)
                np_render = torch.permute(img, (1, 2, 0)).detach().cpu().numpy()
                cv2.imshow(f"rendered{i}", np_render)

            cv2.waitKey(1)

    def GaussianMap(self, mapping_result_instance):
        if not mapping_result_instance[0]:
            return
        mapping_result = mapping_result_instance[1]
        status = mapping_result[0]

        if (not status[0]) and (not status[1]) and (not status[2]):
            return

        self.Flag_GS_Pause = status[4]

        if status[0] or status[1]:
            sensor = mapping_result[1]
            rgb = sensor[0]
            SP_xyz = sensor[1]
            with torch.no_grad():
                xyz_t = torch.from_numpy(SP_xyz).to(self.device)
                pose = mapping_result[2].to(self.device)  # torch.tensor

            if status[0] and status[1] : #First KF
                # First Keyframe
                self.CreateInitialKeyframe(rgb, xyz_t, pose)  # rgb must be numpy (Super pixel)
                self.getNerfppNorm(pose)
            elif status[0] and not (status[1]):
                # ref_3d_list= mapping_result[3]
                # ref_color_list= mapping_result[4]
                self.CreateKeyframe(rgb, xyz_t, pose)
                self.getNerfppNorm(pose)
                # print("GMAP3")
            if status[2]:
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
            self.InsertionOptimize()
            # if status[3]:
            #     self.FullOptimizeGaussian(True)
            #     print("DENSIFICATION!")

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
            if status[3]:
                self.FullOptimizeGaussian(True)
                print("DENSIFICATION!")


        return


