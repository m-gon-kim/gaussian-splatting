
import cv2
import math
import numpy as np
import torch
from torch import nn
class MTFMapper:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.device = "cuda"

        with torch.no_grad():
            self.intr = torch.zeros((3, 3), device=self.device, dtype=torch.float32)
            self.intr_np = np.eye(3, dtype=np.float32)
            self.pointclouds_ptr = torch.empty((1,0), dtype=torch.int32, device=self.device)
            self.pointclouds_desc = torch.empty((0,32), dtype=torch.uint8, device=self.device)
            self.pointclouds = torch.empty((7, 0), dtype=torch.float32, device=self.device)  # xyz, rgb, cntr
            self.Current_gray_gpuMat = cv2.cuda_GpuMat()
            self.KF_poses = torch.empty((4, 4, 0), dtype=torch.float32, device=self.device)
            self.GKF_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            self.GKF_index_list = torch.empty((0), dtype=torch.int32, device=self.device)
            self.frustum_center = torch.zeros((4,1), dtype=torch.float32, device=self.device)
            self.frustum_radius = 1.6

        self.SetIntrinsics()

        # from images
        self.KF_rgb_list = []
        self.KF_xyz_list = []  # Converted from Depth map
        self.KF_superpixel_list = []
        self.KF_bow_list = []
        self.KF_kp_list = []
        self.KF_covis_list = []
        self.KF_loop_list = []
        self.KF_essen_list = []
        self.index_2D_3D = []

        self.orb_cuda = cv2.cuda_ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                            WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )
        self.orb = cv2.ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                  WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )
        self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        self.dictionary = np.loadtxt("dictionary_cuda.txt").astype(np.uint8)
        self.bowDiction = cv2.BOWImgDescriptorExtractor(self.orb,
                                                        cv2.BFMatcher(cv2.NORM_HAMMING))  # ORB 여야 작동함. ORB CUDA는 안됌
        self.bowDiction.setVocabulary(self.dictionary)
        self.KF_orb_list = []


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

        self.intr_np[0][0] = fx
        self.intr_np[0][2] = cx
        self.intr_np[1][1] = fy
        self.intr_np[1][2] = cy
        self.intr_np[2][2] = 1

        self.frustum_center[2][0] = 1.6
        self.frustum_center[3][0] = 1.0




    def CreateInitialKeyframe(self, rgb, KF_xyz, KF_orb, keyframe_pose):
        self.KF_rgb_list.append(rgb)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_orb_list.append(KF_orb)
        with torch.no_grad():
            self.KF_poses = torch.cat((self.KF_poses, keyframe_pose.unsqueeze(dim=2)), dim=2)
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()
        self.index_2D_3D.append(torch.arange(len(self.orb_cuda.convert(KF_orb[0]))).to(self.device))

        self.KF_covis_list.append([])
        self.KF_loop_list.append([])
        self.KF_essen_list.append([])

    def CreateKeyframe(self, rgb, KF_xyz, KF_orb, pose):
        self.KF_rgb_list.append(rgb)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_orb_list.append(KF_orb)
        with torch.no_grad():
            self.KF_poses = torch.cat((self.KF_poses, pose.unsqueeze(dim=2)), dim=2)
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()


    def ConvertCamXYZ2GlobalXYZ(self, cam_xyz, pose):
        with torch.no_grad():
            ones = torch.full((1, cam_xyz.shape[1]), 1.0, dtype=torch.float32, device=self.device)
            cam_xyz_homo = torch.cat((cam_xyz, ones), dim=0)
            print("cam_xyz_homo:",cam_xyz_homo.dtype)
            print("pose:",pose.dtype)
            global_xyz = torch.matmul(pose, cam_xyz_homo)
        return global_xyz
    def CreateInitialPointClouds(self, pose, xyz, rgb, orb):
        keypoints = self.orb_cuda.convert(orb[0])
        des_from_cuda = orb[1]
        descriptors = des_from_cuda.download()
        descriptor_torch = torch.from_numpy(descriptors).to(self.device)  # shape(N x 32)

        cam_xyz_list = []
        cam_rgb_list = []
        for kp in keypoints:
            u_f, v_f = kp.pt
            u = int(u_f)
            v = int(v_f)
            cam_xyz_list.append(xyz[v][u])
            cam_rgb_list.append(rgb[v][u])

        cam_xyz_torch = torch.from_numpy(np.array(cam_xyz_list)).T.to(self.device)
        cam_rgb_torch = torch.from_numpy(np.array(cam_rgb_list)).T.to(self.device)
        global_xyz_torch = self.ConvertCamXYZ2GlobalXYZ(cam_xyz_torch, pose).to(self.device)

        pointclouds = torch.empty((7, cam_xyz_torch.shape[1]), device=self.device)
        pointclouds[:3, :] = global_xyz_torch[:3, :].detach()
        pointclouds[3:6, :] = cam_rgb_torch[:, :].detach()
        pointclouds[6, :] = 2

        pointcloud_ptr = torch.arange(cam_xyz_torch.shape[1], dtype=torch.int32,
                                      device=self.device).unsqueeze(1).T

        self.pointclouds = torch.cat((self.pointclouds, pointclouds), dim=1)
        self.pointclouds_ptr = torch.cat((self.pointclouds_ptr, pointcloud_ptr), dim=1)
        self.pointclouds_desc = torch.cat((self.pointclouds_desc, descriptor_torch), dim=0)

    def ProjectMapToFrame(self, init_pose, orb, xyz, rgb):
        # TODO: 2D-3D correspondence List를 만들어야함. (2D keypoint가 3D point를 참조할 수 있는 index임)

        # 1. PointCloud를 camera space로 변경한다.
        # 2. Far/Near를 crop
        # 3. Proejction 하고 boundary 체크한다.
        # 4. 살아남은 녀석들만, ORB Matching수행한다. 매치된 pointcloud는 2d ORB와 인덱싱한다.
        # 5. Match가 안된 Cam Keypoint는 pointcloud로 만든다. 생성한 pointcloud는 2d ORB와 인덱싱한다.
        # 6. 1000개의 ORB는 corresponding하는 3d pointcloud가 항상 있다. 추후 BA에 사용한다.

        boundary_padding = 300  # pixel

        # 1. PointCloud를 camera space로 변경한다.
        global_xyz = self.pointclouds[:3, :].detach()
        ones = torch.full((1, global_xyz.shape[1]), 1.0, dtype=torch.float32, device=self.device)
        global_xyz_homo = torch.cat((global_xyz, ones), dim=0)
        index_3D = torch.arange(global_xyz.shape[1]).to(self.device)
        projected_xyz = torch.matmul(torch.inverse(init_pose), global_xyz_homo)[:3, :]

        # 1번 과정 완료

        # 2. Far/Near를 crop
        cam_xyz_dist = torch.norm(projected_xyz, dim=0)
        far_mask = cam_xyz_dist < 3.0
        cam_xyz_dist = cam_xyz_dist[far_mask]
        projected_xyz = projected_xyz[:, far_mask]
        pc_desc = self.pointclouds_desc[far_mask, :]
        index_3D = index_3D[far_mask]

        near_mask = cam_xyz_dist > 0.1
        # cam_xyz_dist = cam_xyz_dist[near_mask]
        projected_xyz = projected_xyz[:, near_mask]
        pc_desc = pc_desc[near_mask, :]
        index_3D = index_3D[near_mask]
        # 2번 과정 완료


        # 3. Proejction 하고 boundary 체크한다.
        projected_uv = torch.matmul(self.intr, projected_xyz)
        zero_mask = projected_uv[2, :].ne(0)
        projected_uv_zero_mask = projected_uv[:, zero_mask]
        pc_desc = pc_desc[zero_mask, :]
        index_3D = index_3D[zero_mask]

        projected_uv_zero_mask = projected_uv_zero_mask / projected_uv_zero_mask[2, :]  # projection 한 uv
        u_min_boundary = projected_uv_zero_mask[0, :] > - boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:2, u_min_boundary]
        pc_desc = pc_desc[u_min_boundary, :]
        index_3D = index_3D[u_min_boundary]

        v_min_boundary = projected_uv_zero_mask[1, :] > - boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, v_min_boundary]
        pc_desc = pc_desc[v_min_boundary, :]
        index_3D = index_3D[v_min_boundary]

        u_max_boundary = projected_uv_zero_mask[0, :] < self.width + boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, u_max_boundary]
        pc_desc = pc_desc[u_max_boundary, :]
        index_3D = index_3D[u_max_boundary]

        v_max_boundary = projected_uv_zero_mask[1, :] < self.height + boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, v_max_boundary]
        pc_desc = pc_desc[v_max_boundary, :]
        index_3D = index_3D[v_max_boundary]

        # print(cam_uv_zero_mask.shape, pc_desc.shape, pointclouds_param.shape)
        # 3번 과정 완료

        # 4. 살아남은 녀석들만, ORB Matching수행한다.
        pc_desc_np = cv2.cuda_GpuMat()
        pc_desc_np.upload(pc_desc.detach().cpu().numpy())
        cam_desc_np = orb[1]
        matches = self.bf.match(cam_desc_np, pc_desc_np)
        matches = sorted(matches, key=lambda x: x.distance)
        # 4번 과정 완료


        # 5. Match가 안된 Cam Keypoint는 pointcloud로 만든다.
        cam_kp = self.orb_cuda.convert(orb[0])
        cam_kp_mask = torch.zeros(len(cam_kp), dtype=torch.bool).to(self.device)
        index_2D_3D = torch.full((1, len(cam_kp)), -1, dtype=torch.int32, device=self.device).squeeze()
        for pair in matches:
            if pair.distance < 40:
                diff = projected_uv_zero_mask[:, pair.trainIdx] - torch.tensor([cam_kp[pair.queryIdx].pt], dtype=torch.float32, device=self.device)
                if torch.norm(diff) < 50:
                    # pixel 좌표계 오차가 적을 때만 True (50px 이내)
                    cam_kp_mask[pair.queryIdx] = True  # matching된 uv를 지칭, 1000개다.
                    pc_ptr_index = index_3D[pair.trainIdx] # pointcloud_ptr
                    # self.pointclouds_ptr[0, pc_ptr_index] = pc_ptr_index

                    index_2D_3D[pair.queryIdx] = pc_ptr_index
                    self.pointclouds[6, pc_ptr_index] += 1
            else:
                break
        descriptors = cam_desc_np.download()
        descriptor_list = []

        cam_xyz_list = []
        cam_rgb_list = []
        pc_index = global_xyz.shape[1]
        print("camp_kp_mask", torch.count_nonzero(cam_kp_mask), cam_kp_mask.shape)
        for i in range(cam_kp_mask.shape[0]):
            if not cam_kp_mask[i]:
                index_2D_3D[i] = pc_index
                u, v = cam_kp[i].pt
                cam_xyz_list.append(xyz[int(v)][int(u)])
                cam_rgb_list.append(rgb[int(v)][int(u)])
                descriptor_list.append(descriptors[i])
                pc_index += 1

        cam_xyz_torch = torch.from_numpy(np.array(cam_xyz_list)).T.to(self.device)
        cam_rgb_torch = torch.from_numpy(np.array(cam_rgb_list)).T.to(self.device)
        global_xyz_torch = self.ConvertCamXYZ2GlobalXYZ(cam_xyz_torch, init_pose).to(self.device)

        pointcloud = torch.ones((7, cam_xyz_torch.shape[1]), dtype=torch.float32, device=self.device)
        pointcloud[:3, :] = global_xyz_torch[:3, :].detach()
        pointcloud[3:6, :] = cam_rgb_torch.detach()

        pointcloud_ptr = torch.arange(pointcloud.shape[1], dtype=torch.int32, device=self.device).unsqueeze(1).T
        pointcloud_ptr += int(self.pointclouds.shape[1])

        self.pointclouds = torch.cat((self.pointclouds, pointcloud.detach()), dim=1)
        self.pointclouds_ptr = torch.cat((self.pointclouds_ptr, pointcloud_ptr), dim=1)

        descriptor_torch = torch.tensor(np.array(descriptor_list)).to(self.device)  # shape(N x 32)
        self.pointclouds_desc = torch.cat((self.pointclouds_desc, descriptor_torch), dim=0)


        self.index_2D_3D.append(index_2D_3D)
        # 5번 과정 완료


    def ComputeCovis(self, current_orb, ref_orb):
        result_covis = False
        result_loop = False
        result_essen = False

        current_kp, current_des = current_orb
        ref_kp, ref_des = ref_orb
        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        match_cnt = 0
        covis_threshold = 15
        loop_threshold = 30
        essen_threshold = 100
        for j in matches:
            if j.distance < 40:
                match_cnt += 1
            else:
                break
        if match_cnt < covis_threshold:
            return False, []
        if match_cnt > covis_threshold:
            result_covis = True

        if match_cnt > loop_threshold:
            result_loop = True

        if match_cnt > essen_threshold:
            result_essen = True
        return True, [result_covis, result_loop, result_essen]

    def LoopBuildCovisGraph(self, current_kf, ref_kf):
        current_orb = self.KF_orb_list[current_kf]
        ref_orb = self.KF_orb_list[ref_kf]
        result_covis = self.ComputeCovis(current_orb, ref_orb)
        if not result_covis[0]:
            return
        if result_covis[1][0]:
            if not (current_kf in self.KF_covis_list[ref_kf]):
                self.KF_covis_list[ref_kf].append(current_kf)
            if not (ref_kf in self.KF_covis_list[current_kf]):
                self.KF_covis_list[current_kf].append(ref_kf)
        if result_covis[1][1]:
            if not (current_kf in self.KF_loop_list[ref_kf]):
                self.KF_loop_list[ref_kf].append(current_kf)
            if not (ref_kf in self.KF_loop_list[current_kf]):
                self.KF_loop_list[current_kf].append(ref_kf)
        if result_covis[1][2]:
            if not (current_kf in self.KF_essen_list[ref_kf]):
                self.KF_essen_list[ref_kf].append(current_kf)
            if not (ref_kf in self.KF_essen_list[current_kf]):
                self.KF_essen_list[current_kf].append(ref_kf)


    def BuildCovisGraph(self, current_orb, ref_idx):
        loop_end = 5
        if self.KF_poses.shape[2] < 5:
            loop_end = self.KF_poses.shape[2]  # 현재 포즈가 들어가기 이전. / 이전 포즈가 가장 최신이다.

        covis_list = []
        loop_list = []
        essen_list = []
        current_idx = len(self.KF_covis_list)
        ref_candidate = self.KF_covis_list[ref_idx].copy()
        ref_candidate.append(current_idx-1)

        for ref_idx in ref_candidate:
            ref_orb = self.KF_orb_list[ref_idx]
            result_covis = self.ComputeCovis(current_orb, ref_orb)
            if not result_covis[0]:
                break
            if result_covis[1][0]:
                covis_list.append(ref_idx)
                if not (current_idx in self.KF_covis_list[ref_idx]):
                    self.KF_covis_list[ref_idx].append(current_idx)
            if result_covis[1][1]:
                loop_list.append(ref_idx)
                if not (current_idx in self.KF_loop_list[ref_idx]):
                    self.KF_loop_list[ref_idx].append(current_idx)
            if result_covis[1][2]:
                essen_list.append(ref_idx)
                if not (current_idx in self.KF_essen_list[ref_idx]):
                    self.KF_essen_list[ref_idx].append(current_idx)
        self.KF_covis_list.append(covis_list)
        self.KF_loop_list.append(loop_list)
        self.KF_essen_list.append(essen_list)

    def SimilarityBOW(self, desc1, desc2):
        diff = desc1 / np.linalg.norm(desc1) - desc2 / np.linalg.norm(desc2)
        result = 1 - 0.5 * np.linalg.norm(diff)
        return result

    def LoopMatchLists(self, loop_list, recent_list):
        loop_list.sort()
        recent_list.sort(reverse=True)
        for ref_kf in loop_list:
            for target_kf in recent_list:
                self.UpdatePointcloudsPTRSingle(target_kf, ref_kf)

    def DetectLoopGetOldList(self, current_idx):
        result = False
        bow_desc = self.KF_bow_list[current_idx]
        loop_neighbor_list = self.KF_covis_list[current_idx]
        bow_score_min = 1.0

        # BOW score Minimum 값 (기준) 계산
        for loop_neighbor in loop_neighbor_list:
            bow_neighbor = self.KF_bow_list[loop_neighbor]
            score = self.SimilarityBOW(bow_desc, bow_neighbor)
            if bow_score_min > score:
                bow_score_min = score
        if bow_score_min < 0.67:  # 휴리스틱
            bow_score_min = 0.67

        # loop candidate 확보
        loop_candidate = []
        max_score = -1
        best_kf = -1
        for j in range(len(self.KF_bow_list) - 1):
            bow_candidate = self.KF_bow_list[j]
            score_new = self.SimilarityBOW(bow_desc, bow_candidate)
            if bow_score_min < score_new and (not (j in loop_neighbor_list)):
                # BOW 점수를 만족 and covis list에 없는 프레임만 추가
                loop_candidate.append((j, score_new))
                if max_score < score_new:
                    best_kf = j
                    max_score = score_new
        if len(loop_candidate) < 3:
            # loop candidate 이 3개 미만이면 Loop Detection 종료
            return False, -1, -1

        # loop candidate 중 3개 이상 연속인 경우를 탐색
        loop_candidate.sort()
        cntr_serial = 1
        match2d2d_candidate_list = []
        match2d2d_candidate_list_score = []
        for i in range(1, len(loop_candidate)):
            if loop_candidate[i][0] - loop_candidate[i - 1][0] == 1:
                cntr_serial += 1
            else:
                if cntr_serial > 2:
                    # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
                    result = True
                    for j in range(cntr_serial):
                        hard_close = None
                        ref_kf = loop_candidate[i - j - 1]
                        # ref_idx = loop_candidate[i - j - 1][0]
                        if not (ref_kf[0] in match2d2d_candidate_list):
                            match2d2d_candidate_list_score.append(ref_kf)
                            match2d2d_candidate_list.append(ref_kf[0])
                cntr_serial = 0
        if cntr_serial > 2:
            # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
            result = True
            for j in range(cntr_serial):
                hard_close = None
                ref_kf = loop_candidate[i - j - 1]
                # ref_idx = loop_candidate[- j - 1][0]
                if not (ref_kf[0] in match2d2d_candidate_list):
                    match2d2d_candidate_list_score.append(ref_kf)
                    match2d2d_candidate_list.append(ref_kf[0])

        if not result:
            return False, -1, -1

        else:
            match2d2d_candidate_list_score = sorted(match2d2d_candidate_list_score, key=lambda x: x[1], reverse=True)

            print("match2d2d_candidate_list", current_idx, match2d2d_candidate_list)
            return result, match2d2d_candidate_list, match2d2d_candidate_list_score[0][0]  # , oldest_loop_frame
    def DetectLoop(self, current_idx):
        result = False
        bow_desc = self.KF_bow_list[current_idx]
        loop_neighbor_list = self.KF_covis_list[current_idx]
        bow_score_min = 1.0

        # BOW score Minimum 값 (기준) 계산
        for loop_neighbor in loop_neighbor_list:
            bow_neighbor = self.KF_bow_list[loop_neighbor]
            score = self.SimilarityBOW(bow_desc, bow_neighbor)
            if bow_score_min > score:
                bow_score_min = score
        if bow_score_min < 0.67:  # 휴리스틱
            bow_score_min = 0.67

        # loop candidate 확보
        loop_candidate = []
        for j in range(len(self.KF_bow_list) - 1):
            bow_candidate = self.KF_bow_list[j]
            score_new = self.SimilarityBOW(bow_desc, bow_candidate)
            if bow_score_min < score_new and (not (j in loop_neighbor_list)):
                # BOW 점수를 만족 and covis list에 없는 프레임만 추가
                loop_candidate.append(j)
        if len(loop_candidate) < 3:
            # loop candidate 이 3개 미만이면 Loop Detection 종료
            return False, -1

        # loop candidate 중 3개 이상 연속인 경우를 탐색
        loop_candidate.sort()
        cntr_serial = 1
        match2d2d_candidate_list = []
        for i in range(1, len(loop_candidate)):
            if loop_candidate[i] - loop_candidate[i - 1] == 1:
                cntr_serial += 1
            else:
                if cntr_serial > 2:
                    # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
                    result = True
                    for j in range(cntr_serial):
                        hard_close = None
                        ref_idx = loop_candidate[i - j - 1]
                        if not (ref_idx in match2d2d_candidate_list):
                            match2d2d_candidate_list.append(ref_idx)
                cntr_serial = 0
        if cntr_serial > 2:
            # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
            result = True
            for j in range(cntr_serial):
                hard_close = None
                ref_idx = loop_candidate[- j - 1]
                if not (ref_idx in match2d2d_candidate_list):
                    match2d2d_candidate_list.append(ref_idx)

        if not result:
            return False, -1

        else:
            match2d2d_candidate_list.sort()
            oldest_frame = match2d2d_candidate_list[0]
            return result, oldest_frame  # , oldest_loop_frame
    def DetectLoopOldframe(self, current_idx):
        result = False
        bow_desc = self.KF_bow_list[current_idx]
        loop_neighbor_list = self.KF_covis_list[current_idx]
        bow_score_min = 1.0
        
        # BOW score Minimum 값 (기준) 계산
        for loop_neighbor in loop_neighbor_list:
            bow_neighbor = self.KF_bow_list[loop_neighbor]
            score = self.SimilarityBOW(bow_desc, bow_neighbor)
            if bow_score_min > score:
                bow_score_min = score
        if bow_score_min < 0.67:  # 휴리스틱
            bow_score_min = 0.67
        
        # loop candidate 확보
        loop_candidate = []
        for j in range(len(self.KF_bow_list) - 1):
            bow_candidate = self.KF_bow_list[j]
            score_new = self.SimilarityBOW(bow_desc, bow_candidate)
            if bow_score_min < score_new and (not (j in loop_neighbor_list)):
                # BOW 점수를 만족 and covis list에 없는 프레임만 추가
                loop_candidate.append(j)
        if len(loop_candidate) < 3:
            # loop candidate 이 3개 미만이면 Loop Detection 종료
            return False, -1

        # loop candidate 중 3개 이상 연속인 경우를 탐색
        loop_candidate.sort()
        cntr_serial = 1
        match2d2d_candidate_list = []
        for i in range(1, len(loop_candidate)):
            if loop_candidate[i] - loop_candidate[i - 1] == 1:
                cntr_serial += 1
            else:
                if cntr_serial > 2:
                    # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
                    result = True
                    for j in range(cntr_serial):
                        hard_close = None
                        ref_idx = loop_candidate[i - j - 1]
                        if not (ref_idx in match2d2d_candidate_list):
                            match2d2d_candidate_list.append(ref_idx)
                cntr_serial = 0
        if cntr_serial > 2:
            # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
            result = True
            for j in range(cntr_serial):
                hard_close = None
                ref_idx = loop_candidate[- j - 1]
                if not (ref_idx in match2d2d_candidate_list):
                    match2d2d_candidate_list.append(ref_idx)

        if not result:
            return False, -1

        else:
            match2d2d_candidate_list_tmp = match2d2d_candidate_list.copy()
            for match2d2d_candidate in match2d2d_candidate_list:
                loop_list = self.KF_loop_list[match2d2d_candidate].copy()
                for loop_candidate in loop_list:
                    if not (loop_candidate in match2d2d_candidate_list_tmp):
                        match2d2d_candidate_list_tmp.append(loop_candidate)

            current_candidate_list = self.KF_loop_list[current_idx].copy()
            current_candidate_list.append(current_idx)

            for match2d2d_candidate in match2d2d_candidate_list_tmp:
                for current_candidate in current_candidate_list:
                    self.LoopBuildCovisGraph(current_candidate, match2d2d_candidate)

            current_covis_list = self.KF_covis_list[current_idx].copy()
            current_covis_list.sort()
            return result, current_covis_list[0]  # , oldest_loop_frame

    def LoopCloseHard(self, current_idx, ref_idx):
        ref_xyz = self.KF_xyz_list[ref_idx]
        pnp_ref_3d_list = []
        pnp_query_2d_list = []
        current_kp, current_des = self.KF_orb_list[current_idx]
        current_kp_cpu = self.orb_cuda.convert(current_kp)
        ref_kp, ref_des = self.KF_orb_list[ref_idx]
        ref_kp_cpu = self.orb_cuda.convert(ref_kp)

        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        for pair in matches:
            if pair.distance < 40:
                ref_u, ref_v = ref_kp_cpu[pair.trainIdx].pt
                ref_u = int(ref_u)
                ref_v = int(ref_v)
                current_u, current_v = current_kp_cpu[pair.queryIdx].pt
                current_u = int(current_u)
                current_v = int(current_v)
                # Skip the edge of the images
                if current_u == 0 or current_v == 0 or ref_u == 0 or ref_v == 0 or current_u == self.width - 1 \
                        or current_v == self.height - 1 or ref_u == self.width - 1 or ref_v == self.height - 1:
                    continue

                pnp_ref_3d_list.append(ref_xyz[ref_v][ref_u])
                pnp_query_2d_list.append(np.array([float(current_u), float(current_v)]))
        print("HARD CLOSE CNT",current_idx, ref_idx, len(pnp_ref_3d_list))
        if len(pnp_ref_3d_list) <30:
            return False, None
        ref_3d_list = np.array(pnp_ref_3d_list)
        query_2d_list = np.array(pnp_query_2d_list)

        print("current", current_idx, "ref", ref_idx)
        print("ref_3d_list", ref_3d_list.shape)
        print("query_2d_list", query_2d_list.shape)

        # Crop near points (for PNP Solver)
        z_mask_1 = ref_3d_list[:, 2] > 0.1
        ref_3d_list = ref_3d_list[z_mask_1]
        query_2d_list = query_2d_list[z_mask_1]
        # Crop far points (for PNP Solver)
        z_mask_2 = ref_3d_list[:, 2] <= 3
        ref_3d_list = ref_3d_list[z_mask_2]
        query_2d_list = query_2d_list[z_mask_2]

        # PNP Solver
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(ref_3d_list, query_2d_list, self.intr_np,
                                                      distCoeffs=None, flags=cv2.SOLVEPNP_EPNP,
                                                      confidence=0.9999,
                                                      reprojectionError=1, iterationsCount=1)

        rot, _ = cv2.Rodrigues(rvec)
        with torch.no_grad():
            relative_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            relative_pose[:3, :3] = torch.from_numpy(rot)
            relative_pose[:3, 3] = torch.from_numpy(tvec).squeeze()
            ref_pose = self.KF_poses[:, :, ref_idx]
            current_pose = torch.matmul(ref_pose, torch.inverse(relative_pose))
        return True, current_pose.detach()

    def UpdatePointcloudsPTR(self, current_idx, ref_idx):
        idx_list = self.KF_covis_list[current_idx].copy()
        idx_list.append(current_idx)
        idx_list.sort(reverse = True)

        ref_orb = self.KF_orb_list[ref_idx]
        ref_kp, ref_des = ref_orb
        ref_index = self.index_2D_3D[ref_idx]

        del_position = torch.tensor([0,0,999], dtype=torch.float32, device=self.device)

        for i in idx_list:
            print("UpdatePointCloud", i)
            current_orb = self.KF_orb_list[i]

            current_kp, current_des = current_orb
            matches = self.bf.match(current_des, ref_des)
            matches = sorted(matches, key=lambda x: x.distance)

            current_index = self.index_2D_3D[i]
            for pair in matches:
                if pair.distance < 50:

                    query_ptr_idx = int(current_index[pair.queryIdx])
                    train_ptr_idx = int(ref_index[pair.trainIdx])
                    query_pc_ptr_idx = int(self.pointclouds_ptr[0, query_ptr_idx].detach())
                    train_pc_ptr_idx = int(self.pointclouds_ptr[0, train_ptr_idx].detach())

                    diff = self.pointclouds[:3, query_pc_ptr_idx] - self.pointclouds[:3, train_pc_ptr_idx]
                    if torch.norm(diff) < 1.0 and query_ptr_idx!=train_ptr_idx:
                        if query_pc_ptr_idx > train_pc_ptr_idx:
                            mask = self.pointclouds_ptr[0, :].eq(train_ptr_idx)
                            self.pointclouds_ptr[0, mask] = train_pc_ptr_idx
                            for index_2d_3d in self.index_2D_3D:
                                mask = index_2d_3d.eq(query_ptr_idx)
                                index_2d_3d[mask] = train_ptr_idx

                            self.pointclouds[:3, query_pc_ptr_idx] = del_position.detach()
                            self.pointclouds[6, train_pc_ptr_idx] += self.pointclouds[6, query_pc_ptr_idx].detach()
                        else:
                            mask = self.pointclouds_ptr[0, :].eq(train_pc_ptr_idx)
                            self.pointclouds_ptr[0, mask] = query_pc_ptr_idx
                            for index_2d_3d in self.index_2D_3D:
                                mask = index_2d_3d.eq(train_ptr_idx)
                                index_2d_3d[mask] = query_ptr_idx

                            self.pointclouds[:3, train_pc_ptr_idx] = del_position.detach()
                            self.pointclouds[6, query_pc_ptr_idx] += self.pointclouds[6, train_pc_ptr_idx].detach()
                else:
                    break
    def UpdatePointcloudsPTRSingle(self, current_idx, ref_idx):
        ref_orb = self.KF_orb_list[ref_idx]
        ref_kp, ref_des = ref_orb
        ref_index = self.index_2D_3D[ref_idx]

        del_position = torch.tensor([0,0,999], dtype=torch.float32, device=self.device)

        current_orb = self.KF_orb_list[current_idx]

        current_kp, current_des = current_orb
        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        current_index = self.index_2D_3D[current_idx]
        cntr=0
        for pair in matches:
            if pair.distance < 60:
                cntr+=1
                query_ptr_idx = int(current_index[pair.queryIdx])
                train_ptr_idx = int(ref_index[pair.trainIdx])
                query_pc_ptr_idx = int(self.pointclouds_ptr[0, query_ptr_idx].detach())
                train_pc_ptr_idx = int(self.pointclouds_ptr[0, train_ptr_idx].detach())

                diff = self.pointclouds[:3, query_pc_ptr_idx] - self.pointclouds[:3, train_pc_ptr_idx]
                if torch.norm(diff) < 1.0 and query_ptr_idx != train_ptr_idx:
                    if query_pc_ptr_idx > train_pc_ptr_idx:
                        mask = self.pointclouds_ptr[0, :].eq(train_ptr_idx)
                        self.pointclouds_ptr[0, mask] = train_pc_ptr_idx
                        for index_2d_3d in self.index_2D_3D:
                            mask = index_2d_3d.eq(query_ptr_idx)
                            index_2d_3d[mask] = train_ptr_idx

                        self.pointclouds[:3, query_pc_ptr_idx] = del_position.detach()
                        self.pointclouds[6, train_pc_ptr_idx] += self.pointclouds[6, query_pc_ptr_idx].detach()
                    else:
                        mask = self.pointclouds_ptr[0, :].eq(train_pc_ptr_idx)
                        self.pointclouds_ptr[0, mask] = query_pc_ptr_idx
                        for index_2d_3d in self.index_2D_3D:
                            mask = index_2d_3d.eq(train_ptr_idx)
                            index_2d_3d[mask] = query_ptr_idx

                        self.pointclouds[:3, train_pc_ptr_idx] = del_position.detach()
                        self.pointclouds[6, query_pc_ptr_idx] += self.pointclouds[6, train_pc_ptr_idx].detach()
            else:
                break
        return cntr
    def CheckSuperPixelFrame(self, pose):
        with torch.no_grad():
            trace = torch.matmul(self.GKF_pose, torch.inverse(pose))
            val = float(trace[0][0] + trace[1][1] + trace[2][2])
            if val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0
            angle = math.acos((val-1)/2)

            shift_matrix = self.GKF_pose[:3, 3] - pose[:3, 3]
            shift = torch.dot(shift_matrix, shift_matrix)
        if(angle > 0.5 or shift > 0.5):
            return True
        else:
            return False


    def LoopAdaptiveBA(self, current_idx, iteration_num, point_rate, pose_rate):
        index_2D_3D_all = self.index_2D_3D.copy()

        fixed_frame_list = self.KF_covis_list[current_idx].copy()
        fixed_frame_list.append(current_idx)
        if not (0 in fixed_frame_list):
            fixed_frame_list.append(0)

        for j in range(len(self.KF_bow_list), 0, -1):
            kf_idx = j - 1
            # KF 한칸 씩 전진 한다.
            if kf_idx in fixed_frame_list:
                # fixed pose면 다음으로 넘긴다.
                continue

            # 최신 Frame에서 옛날 방향으로 propagation 한다.
            # 한 프레임 뒤와 현 프레임에서 동시에 발견된 point를 찾는다.
            # 그 point를 기준으로 현프레임 pose를 optimize한다.
            pre_index = index_2D_3D_all[kf_idx +1].detach()
            current_index = index_2D_3D_all[kf_idx].detach()

            pre_expanded = pre_index.unsqueeze(1)
            current_expanded = current_index.unsqueeze(0)

            # Intersection
            # pre_intersection_mask = torch.any(pre_expanded == current_expanded, dim=1)
            current_intersection_mask = torch.any(pre_expanded == current_expanded, dim=0)

            # Keypoints
            # sharing with prev
            print("current_intersection_mask", current_intersection_mask.shape)
            print("self.KF_kp_list[kf_idx].detach()", self.KF_kp_list[kf_idx].detach().shape)
            share_keypoints = self.KF_kp_list[kf_idx].detach()[:, current_intersection_mask]
            share_pointcloud_indice = self.pointclouds_ptr[0, current_index[current_intersection_mask]]  # 1000
            share_pointcloud_seen_from_kf = self.pointclouds[:, share_pointcloud_indice[:]]
            share_cntr_mask = share_pointcloud_seen_from_kf[6, :] > 3  # 1000

            share_pointcloud_seen_from_kf_ctr = share_pointcloud_seen_from_kf[:, share_cntr_mask]
            share_pointcloud_forward = torch.ones((4, share_pointcloud_seen_from_kf_ctr.shape[1]), dtype=torch.float32,
                                                  device=self.device)
            share_pointcloud_forward[:3, :] = share_pointcloud_seen_from_kf_ctr[:3, :]
            share_keypoints = share_keypoints[:, share_cntr_mask]

            # Keypoints
            # Newly found in current
            new_keypoints = self.KF_kp_list[kf_idx].detach()[:, ~current_intersection_mask]
            new_pointcloud_indice = self.pointclouds_ptr[0, current_index[~current_intersection_mask]]  # 1000
            new_pointcloud_seen_from_kf = self.pointclouds[:, new_pointcloud_indice[:]]
            new_cntr_mask = new_pointcloud_seen_from_kf[6, :] > 3  # 1000

            new_pointcloud_seen_from_kf_ctr = new_pointcloud_seen_from_kf[:, new_cntr_mask]
            new_pointcloud_forward = torch.ones((4, new_pointcloud_seen_from_kf_ctr.shape[1]), dtype=torch.float32,
                                                  device=self.device)
            new_pointcloud_forward[:3, :] = new_pointcloud_seen_from_kf_ctr[:3, :]
            new_keypoints = new_keypoints[:, new_cntr_mask]

            # Pose setting
            kf_pose = self.KF_poses[:, :, kf_idx].detach()
            pose_last = torch.eye((4), dtype=torch.float32, device=self.device)
            kf_pose[3:4, :] = pose_last[3:4, :]
            poses_param = nn.Parameter(kf_pose.detach().requires_grad_(True))

            # Pose Optimizer params
            poses_lr = 0.1 ** pose_rate
            l = [
                {'params': [poses_param], 'lr': poses_lr, "name": "poses"}
            ]
            optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)

            # Optimize Pose
            for iteration in range(iteration_num):
                world_to_kf = torch.inverse(poses_param)
                transformed_xyz = torch.matmul(world_to_kf, share_pointcloud_forward)[:3, :]
                projected_uv = torch.matmul(self.intr, transformed_xyz)
                mask = projected_uv[2, :].ne(0)
                projected_uv_mask = projected_uv[:, mask]
                projected_uv_mask = projected_uv_mask / projected_uv_mask[2, :]  # projection 한 uv
                keypoints_masked = share_keypoints[:, mask]

                loss = torch.norm((keypoints_masked - projected_uv_mask[:2, :]), dim=0)
                loss_total = torch.sum(loss) / (projected_uv_mask.shape[1])
                loss_total.backward()
                # loss_total.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Fix pose
            self.KF_poses[:3, :, kf_idx] = poses_param[:3, :].detach()
            fixed_pose = self.KF_poses[:, :, kf_idx].detach()

            # PC Optimizer params
            pointclouds_param = nn.Parameter(new_pointcloud_forward.detach().requires_grad_(True))
            pointclouds_lr = 0.1 ** point_rate
            l = [
                {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            ]
            optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)
            for iteration in range(iteration_num):
                world_to_kf = torch.inverse(fixed_pose)
                transformed_xyz = torch.matmul(world_to_kf, pointclouds_param)[:3, :]
                projected_uv = torch.matmul(self.intr, transformed_xyz)
                mask = projected_uv[2, :].ne(0)
                projected_uv_mask = projected_uv[:, mask]
                projected_uv_mask = projected_uv_mask / projected_uv_mask[2, :]  # projection 한 uv
                keypoints_masked = new_keypoints[:, mask]

                loss = torch.norm((keypoints_masked - projected_uv_mask[:2, :]), dim=0)
                loss_total = torch.sum(loss) / (projected_uv_mask.shape[1])
                loss_total.backward()
                # loss_total.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            #Fix point clouds
            print("Fix pointclouds", (self.pointclouds[:, new_pointcloud_indice[:]]).shape, pointclouds_param.shape)
            (self.pointclouds[:3, new_pointcloud_indice[:]])[:, new_cntr_mask] = pointclouds_param[:3, :]


        GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
        self.GKF_pose = GKF_poses[:, :, -1].detach()


    def LoopBA(self, current_idx, iteration_num, point_rate, pose_rate):
        index_2D_3D_all = self.index_2D_3D.copy()

        covis_list = self.KF_covis_list[current_idx].copy()
        covis_list.append(current_idx)
        covis_list.append(0)

        #Pointcloud Params
        pointclouds = self.pointclouds.detach()
        pointclouds_mask = torch.zeros(pointclouds.shape[1], dtype=torch.bool).to(self.device)
        for kf_idx in covis_list:
            index_2D_3D = index_2D_3D_all[kf_idx]
            #pointer 짓을 해야함.
            print("pointclouds_mask", pointclouds_mask.shape)
            print("self.pointclouds_ptr", self.pointclouds_ptr.shape)
            print("index_2D_3D", index_2D_3D.shape)
            pointclouds_mask[self.pointclouds_ptr[0, index_2D_3D[:]]] = True
        fixed_ponintclouds = pointclouds[:, pointclouds_mask]
        unfixed_ponintclouds = pointclouds[:, ~pointclouds_mask]
        pointclouds_param = nn.Parameter(unfixed_ponintclouds[:3, :].detach().requires_grad_(True))


        #Pose Params
        poses = self.KF_poses.detach()
        pose_mask = torch.zeros(poses.shape[2], dtype=torch.bool).to(self.device)
        fixed_poses_indices = torch.from_numpy(np.array(covis_list)).to(self.device)
        pose_mask[fixed_poses_indices[:]] = True
        fixed_poses = self.KF_poses[:, :, pose_mask].detach()
        unfixed_poses = self.KF_poses[:, :, ~pose_mask].detach()
        poses_param = nn.Parameter(unfixed_poses[:3, :, :].detach().requires_grad_(True))



        #Optimizer params
        pointclouds_lr = 0.1 ** point_rate
        poses_lr = 0.1 ** pose_rate
        l = [
            {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses_param], 'lr': poses_lr, "name": "poses"}
        ]
        optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)

        for iteration in range(iteration_num):
            uv_cnt = 0
            loss_total = 0.0
            print("LoopBA iteration", iteration)

            pointclouds_forward = torch.ones((4, pointclouds.shape[1]), dtype=torch.float32, device=self.device)
            pointclouds_forward[:3, pointclouds_mask] = fixed_ponintclouds[:3, :]
            pointclouds_forward[:3, ~pointclouds_mask] = pointclouds_param

            poses_forward = poses.detach()
            poses_forward[:3, :, pose_mask] = fixed_poses[:3, :, :]
            poses_forward[:3, :, ~pose_mask] = poses_param[:3, :, :]
            pose_last = torch.eye((4), dtype=torch.float32,
                                  device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
            poses_forward[3:4, :, :] = pose_last

            for j in range(len(self.KF_bow_list), 0, -1):
                kf_idx = j - 1
                if kf_idx in covis_list:
                    continue
                keypoints = self.KF_kp_list[kf_idx].detach()

                index_2D_3D = index_2D_3D_all[kf_idx].detach()  # 1000
                pointcloud_indice = self.pointclouds_ptr[0, index_2D_3D[:]]  # 1000
                pointcloud_seen_from_kf = pointclouds_forward[:, pointcloud_indice[:]]
                # pointcloud_seen_from_kf = pointclouds_forward[:, index_2D_3D[0, :]]

                # pointclouds_cntr = torch.index_select(pointclouds[6, :].unsqueeze(dim=0), 1, pointcloud_indice)
                pointclouds_cntr = pointclouds[6, pointcloud_indice[:]]
                print("pointclouds_cntr", pointclouds_cntr.shape)
                cntr_mask = pointclouds_cntr[:] > 3  # 1000

                pointcloud_seen_from_kf_ctr = pointcloud_seen_from_kf[:, cntr_mask]
                keypoints = keypoints[:, cntr_mask]

                pose_from_kf = poses_forward[:, :, kf_idx]
                world_to_kf = torch.inverse(pose_from_kf)

                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf_ctr)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]  # projection 한 uv
                keypoints = keypoints[:, mask]

                loss = torch.norm((keypoints - cam_uv_mask[:2, :]), dim=0)
                loss_total += torch.sum(loss)
                uv_cnt += (cam_uv_mask.shape[1])
                # sorted_loss, _ = loss.sort(dim=0)
                # inlier_num_min = int(loss.shape[0] * 0.0)
                # inlier_num_max = int(loss.shape[0] * 1.0)
                # loss_kf = torch.sum(sorted_loss[inlier_num_min:inlier_num_max])
                # loss_total += loss_kf
                # uv_cnt += (inlier_num_max - inlier_num_min)
            if uv_cnt == 0:
                continue
            loss_total = loss_total / uv_cnt
            # print("LOCAL BA losee:", loss_total)

            print("LoopBA backward A", iteration, loss_total)
            loss_total.backward(retain_graph=True)
            print("LoopBA backward B", iteration, loss_total)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.pointclouds[:3, :] = pointclouds_forward[:3, :].detach()
        self.KF_poses[:3, :, :] = poses_forward[:3, :, :].detach()
        # for i in range(1, len(self.KF_bow_list)):
        #     pose_update = poses_train[:, :, i - 1]
        #     self.KF_poses[:3, :, i] = pose_update.detach()
        GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
        self.GKF_pose = GKF_poses[:, :, -1].detach()
        print("LOOP BA ENDS")

    def LocalBA(self, current_idx, iteration_num, point_rate, pose_rate):
        covis_list = self.KF_covis_list[current_idx].copy()
        if len(covis_list) <3:
            return
        covis_list.append(current_idx)
        covis_list.sort()

        with torch.no_grad():
            pointclouds = self.pointclouds.detach()
            ones = torch.ones((1, pointclouds.shape[1]), dtype=torch.float32, device=self.device)
            BA_poses = torch.empty((3, 4, 0), dtype=torch.float32, device=self.device)
            for neighbor_idx in covis_list:
                neighbor_pose = self.KF_poses[:3, :, neighbor_idx].detach().unsqueeze(dim=2)
                BA_poses = torch.cat((BA_poses, neighbor_pose), dim=2)
            index_2D_3D_all = self.index_2D_3D.copy()
        pointclouds_param = nn.Parameter(torch.cat((pointclouds[:3, :].detach(), ones.detach()), dim=0).requires_grad_(True))
        poses_train = nn.Parameter(BA_poses[:, :, 1:].detach().requires_grad_(True))

        pointclouds_lr =  0.1 ** point_rate
        poses_lr = 0.1 ** pose_rate
        l = [
            {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses_train], 'lr': poses_lr, "name": "poses"}
        ]
        optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)

        for iteration in range(iteration_num):
            uv_cnt = 0
            loss_total = 0.0
            poses = torch.cat((BA_poses[:, :, 0].detach().unsqueeze(dim=2), poses_train), dim=2)
            pose_last = torch.eye((4), dtype=torch.float32,
                                  device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
            poses_four = torch.cat((poses, pose_last), dim=0)
            for i, kf_idx in enumerate(covis_list):
                keypoints = self.KF_kp_list[kf_idx].detach()

                index_2D_3D = index_2D_3D_all[kf_idx].detach()  # 1000
                pointcloud_indice = torch.index_select(self.pointclouds_ptr, 1, index_2D_3D).squeeze()  # 1000
                pointcloud_seen_from_kf = torch.index_select(pointclouds_param, 1, pointcloud_indice)

                pointclouds_cntr = torch.index_select(pointclouds[6, :].unsqueeze(dim=0), 1, pointcloud_indice)
                cntr_mask = pointclouds_cntr[0, :] > 3  # 1000
                pointcloud_seen_from_kf_ctr = pointcloud_seen_from_kf[:, cntr_mask]
                keypoints= keypoints[:, cntr_mask]


                pose_from_kf = poses_four[:, :, i]
                world_to_kf = torch.inverse(pose_from_kf)

                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf_ctr)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]  # projection 한 uv
                keypoints = keypoints[:, mask]

                loss = torch.norm((keypoints - cam_uv_mask[:2, :]), dim=0)
                sorted_loss, _ = loss.sort(dim=0)
                inlier_num_min = int(loss.shape[0] * 0.0)
                inlier_num_max = int(loss.shape[0] * 1.0)
                loss_kf = torch.sum(sorted_loss[inlier_num_min:inlier_num_max])
                loss_total += loss_kf
                uv_cnt += (inlier_num_max - inlier_num_min)
            if uv_cnt == 0:
                continue
            loss_total = loss_total/uv_cnt
            # print("LOCAL BA losee:", loss_total)

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.pointclouds[:3, :] = pointclouds_param[:3, :].detach()

        for i in range(1, len(covis_list)):
            kf_idx = covis_list[i]
            pose_update = poses_train[:, :, i-1]
            self.KF_poses[:3, :, kf_idx] = pose_update.detach()
        GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
        self.GKF_pose = GKF_poses[:, :, -1].detach()
        print("LOOP BA ENDS")

    def FullBA(self, iteration_num, point_rate, pose_rate):
        pointclouds = self.pointclouds.detach()
        ones = torch.ones((1, pointclouds.shape[1]), dtype=torch.float32, device=self.device)

        pointclouds_param = nn.Parameter(
            torch.cat((pointclouds[:3, :].detach(), ones.detach()), dim=0).requires_grad_(True))
        poses_train = nn.Parameter(self.KF_poses[:3, :, 1:].detach().requires_grad_(True))
        index_2D_3D_all = self.index_2D_3D.copy()

        pointclouds_lr = 0.1 ** point_rate
        poses_lr = 0.1 ** pose_rate
        l = [
            {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses_train], 'lr': poses_lr, "name": "poses"}
        ]
        optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)

        for iteration in range(iteration_num):
            uv_cnt = 0
            loss_total = 0.0
            poses = torch.cat((self.KF_poses[:3, :, 0].detach().unsqueeze(dim=2), poses_train), dim=2)
            pose_last = torch.eye((4), dtype=torch.float32,
                                  device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
            poses_four = torch.cat((poses, pose_last), dim=0)
            for j in range(len(self.KF_bow_list), 0, -1):
                i = j-1
                keypoints = self.KF_kp_list[i].detach()

                index_2D_3D = index_2D_3D_all[i].detach()  # 1000
                pointcloud_indice = torch.index_select(self.pointclouds_ptr, 1, index_2D_3D).squeeze()  # 1000
                pointcloud_seen_from_kf = torch.index_select(pointclouds_param, 1, pointcloud_indice)

                pointclouds_cntr = torch.index_select(pointclouds[6, :].unsqueeze(dim=0), 1, pointcloud_indice)
                cntr_mask = pointclouds_cntr[0, :] > 3  # 1000
                pointcloud_seen_from_kf_ctr = pointcloud_seen_from_kf[:, cntr_mask]
                keypoints = keypoints[:, cntr_mask]

                pose_from_kf = poses_four[:, :, i]
                world_to_kf = torch.inverse(pose_from_kf)

                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf_ctr)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]  # projection 한 uv
                keypoints = keypoints[:, mask]

                loss = torch.norm((keypoints - cam_uv_mask[:2, :]), dim=0)
                sorted_loss, _ = loss.sort(dim=0)
                inlier_num_min = int(loss.shape[0] * 0.25)
                inlier_num_max = int(loss.shape[0] * 0.75)
                loss_kf = torch.sum(sorted_loss[inlier_num_min:inlier_num_max])
                loss_total += loss_kf
                uv_cnt += (inlier_num_max - inlier_num_min)
            if uv_cnt == 0:
                continue
            loss_total = loss_total / uv_cnt
            # print("FULL BA losee:", loss_total)

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.pointclouds[:3, :] = pointclouds_param[:3, :].detach()

        for i in range(1, len(self.KF_bow_list)):
            pose_update = poses_train[:, :, i - 1]
            self.KF_poses[:3, :, i] = pose_update.detach()
        GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
        self.GKF_pose = GKF_poses[:, :, -1].detach()
        print("FULL BA ENDS")

    def ResetPointcloudLoop(self, current_frame, oldest_frame):

        current_covis = self.KF_covis_list[current_frame].copy()
        oldest_covis = self.KF_covis_list[oldest_frame].copy()
        oldest_covis.sort()
        current_covis.sort()

        self.KF_loop_list = []
        self.KF_covis_list = []
        self.KF_essen_list = []


        self.pointclouds_ptr = torch.empty((1,0), dtype=torch.int32, device=self.device)
        self.pointclouds_desc = torch.empty((0,32), dtype=torch.uint8, device=self.device)
        self.pointclouds = torch.empty((7, 0), dtype=torch.float32, device=self.device)  # xyz, rgb, cntr


        # initial kf
        kf_pose = self.KF_poses[:, :, 0].detach()
        kf_xyz = self.KF_xyz_list[0]
        kf_rgb = self.KF_rgb_list[0]

        kf_kp, kf_desc = self.KF_orb_list[0]

        self.KF_covis_list.append([])
        self.KF_loop_list.append([])
        self.KF_essen_list.append([])
        self.index_2D_3D = []
        self.index_2D_3D.append(torch.arange(len(self.orb_cuda.convert(kf_kp))).to(self.device).detach())

        self.CreateInitialPointClouds(kf_pose, kf_xyz, kf_rgb, (kf_kp, kf_desc))

        for kf_idx in range(1, self.KF_poses.shape[2]):
            kf_pose = self.KF_poses[:, :, kf_idx].detach()
            kf_xyz = self.KF_xyz_list[kf_idx]
            kf_rgb = self.KF_rgb_list[kf_idx]

            kf_kp, kf_desc = self.KF_orb_list[kf_idx]
            self.BuildCovisGraph(current_orb=(kf_kp, kf_desc), ref_idx = kf_idx-1)
            self.ProjectMapToFrame(kf_pose,(kf_kp, kf_desc), kf_xyz, kf_rgb)


    def ProjectionPropagationCovis(self, idx):
        covis_list = self.KF_covis_list[idx].copy()
        covis_list.sort(reverse=True)
        print("ProjectionPropagationCovis", idx, covis_list)
        for kf in covis_list:
            self.ProjectMapToFrameLoop(kf)

    def LoopCloseHardCovis(self, idx):
        covis_list = self.KF_covis_list[idx].copy()
        covis_list.sort(reverse=True)
        print("CLOSE HARD COVISs", idx, covis_list)
        for kf in covis_list:
            if (kf+1) in covis_list:
                lc_result, loop_pose = self.LoopCloseHard(kf, kf+1)
                if lc_result:
                    self.KF_poses[:, :, kf] = loop_pose.detach()
            else:
                lc_result, loop_pose = self.LoopCloseHard(kf, idx)
                if lc_result:
                    self.KF_poses[:, :, kf] = loop_pose.detach()

    def ProjectionPropagation(self, idx):
        covis_list = self.KF_covis_list[idx].copy()
        del_position = torch.tensor([0,0,999], dtype=torch.float32, device=self.device)
        for kf in covis_list:
            index_dic = self.ProjectMapToFrameLoop(kf)
            # 이제 이 dic에 있는 것들을 전부 del 시키고
            # 갈아낀다
            print("ProjectionPropagation dic begin", kf)
            for k, v in index_dic.items():
                # k = index_2D_3D[pair.queryIdx]
                # v = pc_ptr_index
                # index_2D_3D[pair.queryIdx] = pc_ptr_index 이렇게 바꿔야함
                for index_2D_3D in self.index_2D_3D:
                    mask = index_2D_3D.eq(int(k))
                    index_2D_3D[mask] = int(v)
                    self.pointclouds[:3, k] = del_position.detach()
                    self.pointclouds[6, v] += self.pointclouds[6, k]
            print("ProjectionPropagation dic end", kf)

    def ProjectMapToFrameLoop(self, idx):
        # 1. PointCloud를 camera space로 변경한다.
        # 2. Far/Near를 crop
        # 3. Proejction 하고 boundary 체크한다.
        # 4. 살아남은 녀석들만, ORB Matching수행한다. 매치된 pointcloud는 2d ORB와 인덱싱한다.
        # 5. Match가 안된 Cam Keypoint는 pointcloud로 만든다. 생성한 pointcloud는 2d ORB와 인덱싱한다.
        # 6. 1000개의 ORB는 corresponding하는 3d pointcloud가 항상 있다. 추후 BA에 사용한다.

        init_pose = self.KF_poses[:, :, idx].detach()
        orb = self.KF_orb_list[idx]
        xyz = self.KF_xyz_list[idx]

        boundary_padding = 300  # pixel

        # 1. PointCloud를 camera space로 변경한다.
        global_xyz = self.pointclouds[:3, :].detach()
        ones = torch.full((1, global_xyz.shape[1]), 1.0, dtype=torch.float32, device=self.device)
        global_xyz_homo = torch.cat((global_xyz, ones), dim=0)
        index_3D = torch.arange(global_xyz.shape[1]).to(self.device)
        projected_xyz = torch.matmul(torch.inverse(init_pose), global_xyz_homo)[:3, :]

        # 1번 과정 완료

        # 2. Far/Near를 crop
        cam_xyz_dist = torch.norm(projected_xyz, dim=0)
        far_mask = cam_xyz_dist < 3.0
        cam_xyz_dist = cam_xyz_dist[far_mask]
        projected_xyz = projected_xyz[:, far_mask]
        pc_desc = self.pointclouds_desc[far_mask, :]
        index_3D = index_3D[far_mask]

        near_mask = cam_xyz_dist > 0.1
        # cam_xyz_dist = cam_xyz_dist[near_mask]
        projected_xyz = projected_xyz[:, near_mask]
        pc_desc = pc_desc[near_mask, :]
        index_3D = index_3D[near_mask]
        # 2번 과정 완료

        # 3. Proejction 하고 boundary 체크한다.
        projected_uv = torch.matmul(self.intr, projected_xyz)
        zero_mask = projected_uv[2, :].ne(0)
        projected_uv_zero_mask = projected_uv[:, zero_mask]
        pc_desc = pc_desc[zero_mask, :]
        index_3D = index_3D[zero_mask]

        projected_uv_zero_mask = projected_uv_zero_mask / projected_uv_zero_mask[2, :]  # projection 한 uv
        u_min_boundary = projected_uv_zero_mask[0, :] > - boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:2, u_min_boundary]
        pc_desc = pc_desc[u_min_boundary, :]
        index_3D = index_3D[u_min_boundary]

        v_min_boundary = projected_uv_zero_mask[1, :] > - boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, v_min_boundary]
        pc_desc = pc_desc[v_min_boundary, :]
        index_3D = index_3D[v_min_boundary]

        u_max_boundary = projected_uv_zero_mask[0, :] < self.width + boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, u_max_boundary]
        pc_desc = pc_desc[u_max_boundary, :]
        index_3D = index_3D[u_max_boundary]

        v_max_boundary = projected_uv_zero_mask[1, :] < self.height + boundary_padding
        projected_uv_zero_mask = projected_uv_zero_mask[:, v_max_boundary]
        pc_desc = pc_desc[v_max_boundary, :]
        index_3D = index_3D[v_max_boundary]

        # print(cam_uv_zero_mask.shape, pc_desc.shape, pointclouds_param.shape)
        # 3번 과정 완료

        # 4. 살아남은 녀석들만, ORB Matching수행한다.
        pc_desc_np = cv2.cuda_GpuMat()
        pc_desc_np.upload(pc_desc.detach().cpu().numpy())
        cam_desc_np = orb[1]
        matches = self.bf.match(cam_desc_np, pc_desc_np)
        matches = sorted(matches, key=lambda x: x.distance)
        # 4번 과정 완료

        # 5. Match가 안된 Cam Keypoint는 pointcloud로 만든다.
        cam_kp = self.orb_cuda.convert(orb[0])
        cam_kp_mask = torch.zeros(len(cam_kp), dtype=torch.bool).to(self.device)
        index_2D_3D = self.index_2D_3D[idx]
        del_position = torch.tensor([0,0,999], dtype=torch.float32, device=self.device)

        for pair in matches:
            if pair.distance < 60: # 관대하게 감
                diff = projected_uv_zero_mask[:, pair.trainIdx] - torch.tensor([cam_kp[pair.queryIdx].pt],
                                                                               dtype=torch.float32, device=self.device)
                if torch.norm(diff) < 300:
                    # pixel 좌표계 오차가 적을 때만 True (10px 이내)
                    cam_kp_mask[pair.queryIdx] = True  # matching된 uv를 지칭, 1000개다.

                    pc_ptr_index = index_3D[pair.trainIdx]  # pointcloud_idx
                    index_2d = int(index_2D_3D[pair.queryIdx].detach())

                    if self.pointclouds[6, pc_ptr_index] > 0:
                        index_2D_3D[pair.queryIdx] = pc_ptr_index
                        self.pointclouds_ptr[0, index_2d] = pc_ptr_index
                        self.pointclouds[6, pc_ptr_index] += self.pointclouds[6, index_2d]
                    self.pointclouds[6, index_2d] = -1
                    self.pointclouds[:3, index_2d] = del_position.detach()
            else:
                break


        cam_xyz_list = []
        cam_pc_index_list = []
        for i in range(cam_kp_mask.shape[0]):
            if not cam_kp_mask[i]:
                #xyz를 카메라 포즈 기준으로 옮긴다
                pc_index = index_2D_3D[i].detach()
                while pc_index != self.pointclouds_ptr[0, pc_index]:
                    pc_index = self.pointclouds_ptr[0, pc_index]
                index_2D_3D[i] = pc_index
                cam_pc_index_list.append(pc_index.cpu())
                u, v = cam_kp[i].pt
                cam_xyz_list.append(xyz[int(v)][int(u)])

        cam_xyz_torch = torch.from_numpy(np.array(cam_xyz_list)).T.to(self.device)
        global_xyz_torch = self.ConvertCamXYZ2GlobalXYZ(cam_xyz_torch, init_pose).to(self.device)
        t_list = torch.from_numpy(np.array(cam_pc_index_list))
        self.pointclouds[:3, t_list[:]] = global_xyz_torch[:3, :].detach()



    def UpdatePointcloudsPTRPropagation(self, current_idx, ref_idx):
        ref_orb = self.KF_orb_list[ref_idx]
        ref_kp, ref_des = ref_orb
        ref_index = self.index_2D_3D[ref_idx]

        del_position = torch.tensor([0, 0, 999], dtype=torch.float32, device=self.device)

        current_orb = self.KF_orb_list[current_idx]

        current_kp, current_des = current_orb
        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        current_index = self.index_2D_3D[current_idx]
        cntr = 0
        for pair in matches:
            if pair.distance < 50:
                cntr += 1
                query_ptr_idx = int(current_index[pair.queryIdx])
                train_ptr_idx = int(ref_index[pair.trainIdx])
                query_pc_ptr_idx = int(self.pointclouds_ptr[0, query_ptr_idx].detach())
                train_pc_ptr_idx = int(self.pointclouds_ptr[0, train_ptr_idx].detach())

                diff = self.pointclouds[:3, query_pc_ptr_idx] - self.pointclouds[:3, train_pc_ptr_idx]
                if torch.norm(diff) < 1.0 and query_ptr_idx != train_ptr_idx:
                    mask = self.pointclouds_ptr[0, :].eq(train_ptr_idx)
                    self.pointclouds_ptr[0, mask] = train_pc_ptr_idx
                    for index_2d_3d in self.index_2D_3D:
                        mask = index_2d_3d.eq(query_ptr_idx)
                        index_2d_3d[mask] = train_ptr_idx

                    self.pointclouds[:3, query_pc_ptr_idx] = del_position.detach()
                    self.pointclouds[6, train_pc_ptr_idx] += self.pointclouds[6, query_pc_ptr_idx].detach()

            else:
                break

        return cntr
    def SpatialSearchMerge2(self):
        for i in range(len(self.KF_poses)):
            idx_1 = i
            for j in range(len(self.KF_poses), idx_1+1, -1):
                idx_2 = j - 1
                self.UpdatePointcloudsPTRPropagation(idx_2, idx_1)
    def SpatialSearchMerge(self):
        for i in range(len(self.KF_poses), 0, -1):
            idx_1 = i - 1
            pose1 = self.KF_poses[:, :, idx_1].detach()
            for j in range(idx_1, 0, -1):
                idx_2 = j - 1
                if self.ViewSimilarity(pose1, self.KF_poses[:, :, idx_2].detach()):
                    cntr = self.UpdatePointcloudsPTRSingle(idx_2, idx_1)
                    if cntr > 100:
                        if not (idx_1 in self.KF_essen_list[idx_2]):
                            self.KF_essen_list[idx_2].append(idx_1)
                        if not (idx_2 in self.KF_essen_list[idx_1]):
                            self.KF_essen_list[idx_1].append(idx_2)
                    if cntr > 30:
                        if not (idx_1 in self.KF_loop_list[idx_2]):
                            self.KF_loop_list[idx_2].append(idx_1)
                        if not (idx_2 in self.KF_loop_list[idx_1]):
                            self.KF_loop_list[idx_1].append(idx_2)
                    if cntr > 15:
                        if not (idx_1 in self.KF_covis_list[idx_2]):
                            self.KF_covis_list[idx_2].append(idx_1)
                        if not (idx_2 in self.KF_covis_list[idx_1]):
                            self.KF_covis_list[idx_1].append(idx_2)



    def ViewSimilarity(self, pose1, pose2):
        center1 = torch.matmul(pose1, self.frustum_center).detach()
        center2 = torch.matmul(pose2, self.frustum_center).detach()
        diff = torch.norm(center1 - center2)
        if diff < 2.0:
            return True
        else:
            return False

    def FullBACall(self):
        if len(self.KF_bow_list) <10:
            return  [False, False, False, False], [], [], []
        self.FullBA(10, point_rate=5, pose_rate=4)
        GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
        return [False, False, True, False], [], [], GKF_poses.detach().cpu()

    def Map(self, tracking_result_instance):
        Flag_GMapping = False
        Flag_First_KF = False
        Flag_mid_BA = False
        Flag_BA = False
        Flag_densification = False

        if not tracking_result_instance[0]:  # Abort (System is not awake)
            return [Flag_GMapping, Flag_First_KF, Flag_BA, Flag_densification], []

        tracking_result = tracking_result_instance[1]
        status = tracking_result[0]  #[Tracking Success, First KF]
        sensor = tracking_result[1]
        rgb_img = sensor[0]
        gray_img = sensor[1]
        KF_xyz = sensor[2]

        self.Current_gray_gpuMat.upload(gray_img)
        current_kp, current_des = self.orb_cuda.detectAndComputeAsync(self.Current_gray_gpuMat, None)
        keypoint_list = []
        keypoints = self.orb_cuda.convert(current_kp)
        for kp in keypoints:
            u, v = kp.pt
            keypoint_list.append((int(u), int(v)))
        self.KF_kp_list.append(torch.permute((torch.tensor(np.array(keypoint_list))).to(self.device), (1, 0)))

        bow_desc = self.bowDiction.compute(gray_img, self.orb_cuda.convert(current_kp))
        self.KF_bow_list.append(bow_desc)

        if status[1]:  # First KF
            Flag_GMapping = True
            Flag_First_KF = True
            Flag_mid_BA = False
            Flag_BA = False
            Flag_densification = False
            with torch.no_grad():

                self.CreateInitialKeyframe(rgb_img, KF_xyz, (current_kp, current_des),
                                           torch.eye(4, dtype=torch.float32, device=self.device))
                self.CreateInitialPointClouds(self.KF_poses[:, :, -1].detach(), KF_xyz, rgb_img, (current_kp, current_des))
                self.GKF_index_list = torch.cat((self.GKF_index_list, torch.tensor([0], dtype=torch.int32, device=self.device)), dim=0)
            return [Flag_GMapping, Flag_First_KF, Flag_BA, Flag_densification], [rgb_img, KF_xyz], \
                torch.eye(4, dtype=torch.float32).cpu()

        else:  # Not first KF
            Flag_GMapping = False
            Flag_First_KF = False
            Flag_mid_BA = False
            Flag_BA = False
            Flag_densification = False
            # 이전 키프레임을 기준으로 한 point들을 저장한다.
            # 현재 키프레임과 이전 키프레임 사이에서 생성된 point들인데, origin은 이전 것을 기준으로 함.
            ref_3d_list = tracking_result[3]  # 이전 프레임의, Camera 스페이스 xyz임. / 가우시안에 corner점도 추가할 까 해서 넣은 것
            ref_color_list = tracking_result[4]  # 위에 꺼랑 매칭 되는 RGB임


            # 현 키프레임 이미지와, 새로운 pose를 저장 한다.
            relative_pose = tracking_result[2]
            rot = relative_pose[0]
            quat = relative_pose[1]
            tvec = relative_pose[2]

            with torch.no_grad():
                KF_relative_pose = torch.eye(4, dtype=torch.float32, device=self.device)
                KF_relative_pose[:3, :3] = torch.from_numpy(rot)
                KF_relative_pose[:3, 3] = torch.from_numpy(tvec).squeeze()

                Current_pose = torch.matmul(self.KF_poses[:, :, -1].detach(), torch.inverse(KF_relative_pose))
                self.CreateKeyframe(rgb_img, KF_xyz, (current_kp, current_des), Current_pose)
                self.BuildCovisGraph(current_orb=(current_kp, current_des), ref_idx=len(self.KF_covis_list)-1)
            detect_result, loop_list, best_kf = self.DetectLoopGetOldList(len(self.KF_bow_list)-1)
            if detect_result:
                print('loop detected...')
                loop_list_copy = loop_list.copy()
                loop_list_copy.sort()
                lc_result, loop_pose = self.LoopCloseHard(len(self.KF_bow_list)-1, best_kf)
                if lc_result:
                    self.KF_poses[:, :, len(self.KF_bow_list)-1] = loop_pose.detach()
                    self.ProjectMapToFrame(loop_pose.detach(),(current_kp, current_des), KF_xyz, rgb_img)
                    self.LoopCloseHardCovis(len(self.KF_bow_list)-1)

                    ## 연쇄 Projection 수행 해야함
                    self.ProjectionPropagationCovis(len(self.KF_bow_list)-1)

                    #Do prjoections
                    # self.ProjectionPropagation(len(self.KF_bow_list)-1)
                    self.LoopAdaptiveBA(len(self.KF_bow_list) - 1, 10, point_rate=3, pose_rate=3)
                    # self.FullBA(10, point_rate=6, pose_rate=3)

                    # self.UpdatePointcloudsPTRSingle(len(self.KF_bow_list)-1, oldest_frame)
                    # self.SpatialSearchMerge2()
                    # self.UpdatePointcloudsPTR(len(self.KF_bow_list)-1, oldest_frame)  # update point ptr
                    # self.FullBA(10, point_rate=6, pose_rate=3)
                    # self.ResetPointcloudLoop(len(self.KF_bow_list)-1, 0)
                    # self.LoopBA(len(self.KF_bow_list) - 1, 10, point_rate=3, pose_rate=3)
                    # self.FullBA(10, point_rate=3, pose_rate=3)
                # self.SpatialSearchMerge2()
                # self.LoopBA(len(self.KF_bow_list) - 1, 30, point_rate=3, pose_rate=2)
                # self.FullBA(10, point_rate=3, pose_rate=2)

                    Flag_densification = True
                    Flag_BA = True
                else:
                    self.ProjectMapToFrame(Current_pose,(current_kp, current_des), KF_xyz, rgb_img)
                    self.LocalBA(len(self.KF_bow_list) - 1, 10, point_rate=5, pose_rate=4)
                    Flag_BA = True
            else:
                self.ProjectMapToFrame(Current_pose,(current_kp, current_des), KF_xyz, rgb_img)
                self.LocalBA(len(self.KF_bow_list) - 1, 10, point_rate=5, pose_rate=4)
                Flag_BA = True

            print("kf: ", len(self.KF_bow_list)-1)
            # GaussianKF selection
            with torch.no_grad():
                if self.CheckSuperPixelFrame(self.KF_poses[:, :, -1]):
                    self.GKF_pose = self.KF_poses[:, :, -1].detach()
                    self.GKF_index_list = torch.cat(
                        (self.GKF_index_list, torch.tensor([self.KF_poses.shape[2] - 1], dtype=torch.int32,
                                                           device=self.device)), dim=0)
                    GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
                    Flag_GMapping = True

                    return [Flag_GMapping, Flag_First_KF, Flag_BA, Flag_densification], [rgb_img, KF_xyz], Current_pose.detach().cpu(), GKF_poses.detach().cpu()
                else:
                    GKF_poses = torch.index_select(self.KF_poses, 2, self.GKF_index_list)
                    return [Flag_GMapping, Flag_First_KF, Flag_BA, Flag_densification], [], [], GKF_poses.detach().cpu()


