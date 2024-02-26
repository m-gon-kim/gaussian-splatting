import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv

from utility import Rot2Quat, QuaternionInfo

class TrackerTorch:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.device = "cuda"
        self.xy_one = None
        self.orb = None
        self.projection_matrix = None
        self.SetORBSettings()
        self.Initial = True
        self.KF_rgb = None
        self.KF_gray = None
        self.KF_gray_gpuMat = cv2.cuda_GpuMat()
        self.Current_gray_gpuMat = cv2.cuda_GpuMat()
        self.KF_xyz = None
        self.KF_orb = None
        self.intr = np.eye(3, dtype=np.float32)
        with torch.no_grad():
            self.inv_intr = torch.zeros((3, 3), dtype=torch.float32, device=self.device)

        self.SetIntrinsics()
        self.GenerateUVTensor()
        # self.KF_pose = None

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
            nfeatures=40000,
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
            scale_factor = 5000.0

            d = query_kf.unsqueeze(dim=2)
            d = d / scale_factor

            xyz = torch.mul(self.xy_one.detach(), d)
        return xyz

    def CreateInitalKeyframe(self, rgb, depth, orb):
        # Convert Depth to XYZ
        with torch.no_grad():
            KF_depth_map = torch.from_numpy(np.array(depth, dtype=np.float32)).to(self.device)
            KF_xyz = self.RecoverXYZFromKeyFrame(KF_depth_map).detach().cpu().numpy()

        self.KF_rgb = rgb
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()
        self.KF_xyz = KF_xyz
        self.KF_orb = orb
        self.Initial = False

    def CreateKeyframe(self, rgb, depth, orb):
        # Convert Depth to XYZ
        with torch.no_grad():
            KF_depth_map = torch.from_numpy(np.array(depth, dtype=np.float32)).to(self.device)
            KF_xyz = self.RecoverXYZFromKeyFrame(KF_depth_map).detach().cpu().numpy()

        self.KF_rgb = rgb
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()
        self.KF_xyz = KF_xyz
        self.KF_orb = orb

    def Match2D2D(self, orb):
        matches = self.bf.match(orb[1], self.KF_orb[1])
        matches = sorted(matches, key=lambda x: x.distance)

        kf_kp_cpu = self.orb.convert(self.KF_orb[0])
        current_kp_cpu = self.orb.convert(orb[0])

        match_cnt = 0
        match_cnt_threshold = 100
        for j in matches:
            if j.distance < 40:
                match_cnt += 1
            else:
                break
            if match_cnt > 100:
                break

        query_2d_list = []
        ref_3d_list = []
        ref_color_list = []
        for j in matches[:match_cnt]:
            # Append KF lists
            kf_idx = j.trainIdx  # i.trainIdx
            kf_x, kf_y = kf_kp_cpu[kf_idx].pt
            int_kf_x = int(kf_x)
            int_kf_y = int(kf_y)
            if (int_kf_y == 0 or int_kf_x == 0 or int_kf_y == 479 or int_kf_x == 639):
                continue
            ref_3d_list.append(self.KF_xyz[int_kf_y][int_kf_x])
            ref_color_list.append(self.KF_rgb[int_kf_y][int_kf_x])

            # Append Query list
            q_idx = j.queryIdx  # i.trainIdx
            x, y = current_kp_cpu[q_idx].pt
            query_2d_list.append(np.array([x, y], dtype=np.float32))

        return np.array(query_2d_list), np.array(ref_3d_list), np.array(ref_color_list, dtype=np.float32)

    def Track(self, play_instance):
        sensor = play_instance[1]
        rgb = sensor[0]
        gray = sensor[1]
        depth = sensor[2]
        self.Current_gray_gpuMat.upload(gray)
        current_kp, current_des = self.orb.detectAndComputeAsync(self.Current_gray_gpuMat, None)



        # cv2.imshow("Tracking input", rgb)
        # cv2.waitKey(1)

        if not play_instance[0]:  # Abort (System is not awake)
            return

        if self.Initial:
            # initial KF
            self.CreateInitalKeyframe(rgb, depth, (current_kp, current_des))
            # 0.0. Status
            # 0.1. First KF
            return [True, True], [rgb, gray, self.KF_xyz]
        else:
            # Perform 2D-2D matching and, get corresponding 3D(xyz) points
            query_2d_list, ref_3d_list, ref_color_list = self.Match2D2D((current_kp, current_des))

            # Crop near points (for mapping)
            z_mask_0 = ref_3d_list[:, 2] > 0.2
            ref_3d_list = ref_3d_list[z_mask_0]
            ref_color_list = ref_color_list[z_mask_0]
            query_2d_list = query_2d_list[z_mask_0]

            pnp_ref_3d_list = np.copy(ref_3d_list)
            pnp_query_2d_list = np.copy(query_2d_list)

            # Crop near points (for PNP Solver)
            z_mask_1 = pnp_ref_3d_list[:, 2] > 0.5
            pnp_ref_3d_list = pnp_ref_3d_list[z_mask_1]
            pnp_query_2d_list = pnp_query_2d_list[z_mask_1]
            # Crop far points (for PNP Solver)
            z_mask_2 = pnp_ref_3d_list[:, 2] <= 3
            pnp_ref_3d_list = pnp_ref_3d_list[z_mask_2]
            pnp_query_2d_list = pnp_query_2d_list[z_mask_2]

            # print("Track PNP", pnp_query_2d_list.shape, pnp_ref_3d_list.shape )
            # PNP Solver
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_ref_3d_list, pnp_query_2d_list, self.intr,
                                                          distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999,
                                                          reprojectionError=1, iterationsCount=1000)
            rot, _ = cv2.Rodrigues(rvec)
            quat = Rot2Quat(rot)
            # axis, angle = QuaternionInfo(quat)
            shift = np.linalg.norm(tvec[:3, 0].T)

            val = float(rot[0][0] + rot[1][1] + rot[2][2])
            if val > 3.0:
                val = 3.0
            elif val < -1.0:
                val = -1.0
            angle = math.acos((val - 1) * 0.5)

            # print(f"angle: {angle}, shift: {shift}")
            if 0.1 <= angle or 0.3 <= shift :  # Mapping is required
                # print(f"Make KF! angle: {angle}, shift: {shift}")
                self.CreateKeyframe(rgb, depth, (current_kp, current_des))
                relative_pose = [rot, quat, tvec]

                # render_pkg = render(viewpoint_cam, self.gaussian, pipe, bg)
                return [True, False], [rgb, gray, self.KF_xyz], relative_pose, ref_3d_list, ref_color_list
            else:  # Mapping is not required
                return [False], []
