import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv
from superpixel import SuperPixelManager


from utility import Rot2Quat, QuaternionInfo

class TrackerUnreal:
    def __init__(self, dataset, parameters):
        self.width = 640
        self.height = 480
        self.device = "cuda"
        self.xy_one = None
        self.orb = None
        self.projection_matrix = None
        self.KF_gray_gpuMat = cv2.cuda_GpuMat()
        self.Current_gray_gpuMat = cv2.cuda_GpuMat()

        self.intr = np.eye(3, dtype=np.float32)
        self.intr_t = torch.eye(3, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.inv_intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            self.SetIntrinsics(dataset)
        self.GenerateUVTensor()

        self.orb_nfeatures = parameters["tracking"]["orb_nfeatures"]
        self.SetORBSettings()

        self.SP_KF_list = []
        self.SP_KF_rgb_list = []
        self.SP_KF_xyz_list = []
        self.SP_pose_list = []
        self.SP_KF_sp_list = []
        self.KF_num = 0

        self.prev_inv_pose = torch.eye(4, dtype=torch.float32).to(self.device)
        self.sp_manager = SuperPixelManager(self.width, self.height, parameters["gaussian"]["superpixel_creation"])

        self.kf_selection_angle = parameters["tracking"]["kf_selection"]["angle"]
        self.kf_selection_shift = parameters["tracking"]["kf_selection"]["shift"]




    def SetIntrinsics(self, dataset):
        fx, fy, cx, cy = dataset.get_camera_intrinsic()

        self.intr[0][0] = fx
        self.intr[0][2] = cx
        self.intr[1][1] = fy
        self.intr[1][2] = cy
        self.intr[2][2] = 1

        self.intr_t[0][0] = fx
        self.intr_t[0][2] = cx
        self.intr_t[1][1] = fy
        self.intr_t[1][2] = cy
        self.intr_t[2][2] = 1

        self.inv_intr[0][0] = 1 / fx
        self.inv_intr[0][2] = -cx / fx
        self.inv_intr[1][1] = 1 / fy
        self.inv_intr[1][2] = -cy / fy
        self.inv_intr[2][2] = 1

        FoVx = 2*math.atan(640/(2*fx))
        FoVy = 2*math.atan(480/(2*fy))
        with torch.no_grad():
            self.projection_matrix = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).type(torch.FloatTensor).to(self.device)

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        with torch.no_grad():
            P = torch.zeros(4, 4, dtype = torch.float32, device=self.device)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def SetORBSettings(self):
        self.orb=cv2.cuda_ORB.create(
            nfeatures=self.orb_nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,)
        self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)

    def GenerateUVTensor(self):
        with torch.no_grad():
            u = torch.arange(self.width, dtype=torch.float32)
            for i in range(self.height - 1):
                u = torch.vstack((u, torch.arange(self.width)))

            v = torch.tile(torch.arange(self.height), (1, 1)).T
            for i in range(self.width - 1):
                v = torch.hstack((v, torch.tile(torch.arange(self.height, dtype=torch.float32), (1, 1)).T))

            uv = torch.stack((u, v), dim=2).to(self.device)

            ones = torch.ones((uv.shape[0], uv.shape[1], 1), dtype=torch.float32).to(self.device)

            uv_one = torch.cat((uv, ones), dim=2).to(self.device)
            uv_one = torch.unsqueeze(uv_one, dim=2)

            self.xy_one = torch.tensordot(uv_one, self.inv_intr, dims=([3], [1])).squeeze()


    def RecoverXYZFromKeyFrame(self, query_kf):
        with torch.no_grad():
            d = query_kf.unsqueeze(dim=2)
            xyz = torch.mul(self.xy_one.detach(), d)
        return xyz


    def SelectKF(self, play_instance):
        sensor = play_instance[1]
        rgb = sensor[0]  # np array
        # gray = sensor[1]
        depth = sensor[2]  # np array
        pose = torch.from_numpy(sensor[3]).detach().to(torch.float32).to(self.device)
        rgb_torch = torch.from_numpy(rgb).detach().to(self.device)

        Add_Flag = False
        if len(self.SP_pose_list) == 0:
            Add_Flag = True

        else:
            relative_pose = torch.matmul(self.prev_inv_pose.detach(), pose.detach())
            relative_pose = relative_pose / relative_pose[3, 3]
            val = float(relative_pose[0][0] + relative_pose[1][1] + relative_pose[2][2])
            if val > 3.0:
                val = 3.0
            elif val < -1.0:
                val = -1.0
            angle = math.acos((val - 1) * 0.5)
            shift = torch.norm(relative_pose[:3, 3])
            if self.kf_selection_angle <= angle or self.kf_selection_shift <= shift:
                Add_Flag = True
        masked_xyz = None
        masked_rgb = None
        if Add_Flag:
            self.SP_KF_list.append(self.KF_num)
            self.SP_KF_rgb_list.append(rgb)

            # depth to xyz
            KF_depth_map = torch.from_numpy(np.array(depth, dtype=np.float32)).to(self.device)
            KF_xyz = self.RecoverXYZFromKeyFrame(KF_depth_map)
            self.SP_KF_xyz_list.append(KF_xyz.detach())

            self.SP_pose_list.append(pose.detach())
            self.prev_inv_pose = torch.inverse(pose.detach()).detach()

            # Super pixel calculation
            super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
            masked_xyz = KF_xyz[super_pixel_index[0, :], super_pixel_index[1, :], :]
            masked_rgb = rgb_torch[super_pixel_index[0, :], super_pixel_index[1, :], :]

        self.KF_num += 1

        if Add_Flag:
            return [True, Add_Flag], [rgb, masked_xyz.detach().cpu(), masked_rgb.detach().cpu(), pose.detach().cpu(),
                                      self.KF_num]
        else:
            return [True, Add_Flag], [rgb, [], [], pose.detach().cpu(), self.KF_num]
