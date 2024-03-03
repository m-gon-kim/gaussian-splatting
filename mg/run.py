import cv2

from dataset import Dataset

# from tracking import Tracker
from tracking_torch import TrackerTorch
from mapping import Mapper
from mtf_mapping import MTFMapper
from gaussian_mapping import GaussianMapper
from tracking_unreal import TrackerUnreal
from tracking_unreal_all_frames import TrackerUnrealAllFrames
from gaussian_mapping_unreal import GaussianMapperUnreal
from gaussian_mapping_unreal_all_frames import GaussianMapperUnrealAllFrames
from tracking_bypass import TrackerByPass
from gaussian_mapping_bypass import GaussianMapperByPass
import torch.multiprocessing as mp
import time

def PlayDataset(dataset, img_pair_q):
    begin_index = 1
    cnt = dataset.get_data_len()
    awake = True
    for index in range(cnt):
        rgb, gray, d, pose = dataset.ReturnData(index + begin_index)
        img_pair_q.put([awake, [rgb, gray, d, pose]])
    img_pair_q.put([False, []])

def TrackingUnrealTimeAllFrames(dataset, parameters, img_pair_q, tracking_result_q):
    tracker = TrackerUnrealAllFrames(dataset, parameters)
    while True:
        if not img_pair_q.empty():
            instance = img_pair_q.get()
            if not instance[0]:  # Abort (System is not awake)
                print("Tracking Abort")
                tracking_result_q.put([False, []])
                return
            tracking_result = tracker.SelectKF(instance)
            if tracking_result[0][1]:
                tracking_result_q.put([True, tracking_result])

def GaussianMappingUnrealTimeAllFrames(dataset, parameters, tracking_result_q):
    gaussian_mapper = GaussianMapperUnrealAllFrames(dataset, parameters)

    while True:
        if not tracking_result_q.empty():
            # q_size = img_pair_q.qsize()
            # print(f"PROCESS: G-MAPPING Q {q_size}")
            instance = tracking_result_q.get()
            if instance[0]:
                gaussian_mapper.AddGaussianFrame(instance[1])
            else:  # Abort (System is not awake)
                start_time = time.time()
                gaussian_mapper.FullOptimizeGaussian(30000)
                end_time = time.time()
                print("Elapsed time:", end_time - start_time, "seconds")
                gaussian_mapper.Evalulate()
                return


def TrackingUnrealTime(dataset, parameters, img_pair_q, tracking_result_q):
    tracker = TrackerUnreal(dataset, parameters)
    while True:
        if not img_pair_q.empty():
            instance = img_pair_q.get()
            if not instance[0]:  # Abort (System is not awake)
                print("Tracking Abort")
                tracking_result_q.put([False, []])
                return
            tracking_result = tracker.SelectKF(instance)
            tracking_result_q.put([True, tracking_result])


def GaussianMappingUnrealTime(dataset, parameters, tracking_result_q):
    gaussian_mapper = GaussianMapperUnreal(dataset, parameters)

    while True:
        if not tracking_result_q.empty():
            # q_size = img_pair_q.qsize()
            # print(f"PROCESS: G-MAPPING Q {q_size}")
            instance = tracking_result_q.get()
            if instance[0]:
                gaussian_mapper.AddGaussianFrame(instance[1])
            else:  # Abort (System is not awake)
                start_time = time.time()
                gaussian_mapper.FullOptimizeGaussian()
                end_time = time.time()
                print("Elapsed time:", end_time - start_time, "seconds")
                gaussian_mapper.Evalulate()
                return


def TrackingByPass(dataset, parameters, img_pair_q, tracking_result_q):
    tracker = TrackerByPass(dataset, parameters)
    while True:
        if not img_pair_q.empty():
            instance = img_pair_q.get()
            if not instance[0]:  # Abort (System is not awake)
                print("Tracking Abort")
                tracking_result_q.put([False, []])
                return
            tracking_result = tracker.SelectKF(instance)
            if tracking_result[0][1]:
                tracking_result_q.put([True, tracking_result])

def GaussianMappingByPass(dataset, parameters, mapping_result_q):
    gaussian_mapper = GaussianMapperByPass(dataset, parameters)
    while True:
        if not mapping_result_q.empty():
            instance = mapping_result_q.get()
            if instance[0]:
                gaussian_mapper.AddGaussianFrame(instance[1])
                gaussian_mapper.InsertionOptimize()
                gaussian_mapper.FullOptimizeGaussian(10, False)
            else:
                gaussian_mapper.FullOptimizeGaussian(20, False)
                gaussian_mapper.Visualize()
                cv2.waitKey(0)
                return
        gaussian_mapper.Visualize()

def TrackingTorch(dataset, parameters, img_pair_q, tracking_result_q):
    tracker = TrackerTorch(dataset, parameters)

    frame = 0

    awake = True
    while True:
        if not img_pair_q.empty():          
            frame += 1
            instance = img_pair_q.get()
            # print("Tracking frame: ", frame)
            if not instance[0]:  # Abort (System is not awake)
                print("Tracking Abort")
                awake = False
                tracking_result_q.put([awake, []])
                return
            tracking_result = tracker.Track(instance)
            if tracking_result[0][0]:  # Mapping is required
                tracking_result_q.put([awake, tracking_result])

def MTF_Mapping(dataset, parameters, tracking_result_q, mapping_result_q):
    mapper = MTFMapper(dataset, parameters)

    while True:
        if not tracking_result_q.empty():
            q_size= tracking_result_q.qsize()
            # print(f"PROCESS: MAPPING Q {q_size}")
            instance = tracking_result_q.get()
            if not instance[0]:  # Abort (System is not awake)
                print("Mapping Abort")
                # mapper.FullBundleAdjustment()
                mapping_result_q.put([instance[0], []])
                return
            mapping_result = mapper.Map(instance)
            mapping_result_q.put([True, mapping_result])
            if mapping_result[0][4]:
                # loop closing 수행
                loop_close_result = mapper.CloseLoop(mapping_result[4], mapping_result[1][2])
                mapping_result_q.put([True, loop_close_result])
                # mapper.PointPtrUpdate()

def GaussianMappingTest(dataset, parameters, mapping_result_q):
    gaussian_mapper = GaussianMapper(dataset, parameters)

    opt_iter = 0
    viz_iter = 0
    while True:
        if not mapping_result_q.empty():
            q_size = mapping_result_q.qsize()
            # print(f"PROCESS: G-MAPPING Q {q_size}")
            instance = mapping_result_q.get()
            if not instance[0]:  # Abort (System is not awake)
                print("Gaussian Mapping Abort")
                return
            gaussian_mapper.GaussianMap(instance)
            opt_iter+=1
            if opt_iter > 5 and not(instance[1][0][3]):
                gaussian_mapper.OptimizeGaussian(False)
                opt_iter = 0
        else:
            gaussian_mapper.OptimizeGaussian(False)

        # else:
        # gaussian_mapper.OptimizeGaussian()
        gaussian_mapper.Visualize()
        # viz_iter+=1
        # if viz_iter > 5:
        #     viz_iter = 0
        # gaussian_mapper.OptimizeGaussian()


if __name__ == '__main__':
    img_pair_q = mp.Queue()
    tracking_result_q = mp.Queue()
    mapping_result_q = mp.Queue()

    dataset_class = Dataset()
    dataset = dataset_class.GetDataset()
    parameters = dataset_class.parameters

    ####################################################################################################
    ##  Process 생성하기  ################################################################################
    ####################################################################################################

    process_play_data = mp.Process(target=PlayDataset, args=(dataset, img_pair_q,))
    # process_tracking_torch = mp.Process(target=TrackingTorch, args=(dataset, parameters["tracking"], img_pair_q, tracking_result_q,))
    # process_mapping = mp.Process(target=MTF_Mapping, args=(dataset, parameters["mapping"], tracking_result_q, mapping_result_q,))
    # process_gaussian_mapping = mp.Process(target=GaussianMappingTest, args=(dataset, parameters["gaussian"], mapping_result_q,))

    # A. 비 실시간 테스트 (모든 frame)
    # process_tracking_unreal_all_frames = mp.Process(target=TrackingUnrealTimeAllFrames, args=(dataset, parameters, img_pair_q, tracking_result_q,))
    # process_gaussian_mapping_unreal_all_frames = mp.Process(target=GaussianMappingUnrealTimeAllFrames, args=(dataset, parameters["gaussian"], tracking_result_q,))

    # B. 비 실시간 테스트 (Keyframe selection)
    process_tracking_unreal = mp.Process(target=TrackingUnrealTime, args=(dataset, parameters, img_pair_q, tracking_result_q,))
    process_gaussian_mapping_unreal = mp.Process(target=GaussianMappingUnrealTime, args=(dataset, parameters["gaussian"], tracking_result_q,))

    # C. ByPass 테스트
    # process_tracking_bypass = mp.Process(target=TrackingByPass, args=(dataset, parameters, img_pair_q, tracking_result_q,))
    # process_gaussian_mapping_bypass = mp.Process(target=GaussianMappingByPass, args=(dataset, parameters["gaussian"], tracking_result_q,))

    ####################################################################################################
    ##  Process 켜기  ###################################################################################
    ####################################################################################################

    # A. 비 실시간 테스트 (모든 frame)
    # process_gaussian_mapping_unreal_all_frames.start()
    # process_tracking_unreal_all_frames.start()

    # B. 비 실시간 테스트 (Keyframe selection)
    process_gaussian_mapping_unreal.start()
    process_tracking_unreal.start()

    # C. ByPass 테스트
    # process_gaussian_mapping_bypass.start()
    # process_tracking_bypass.start()


    # process_gaussian_mapping.start()
    # process_mapping.start()
    # process_tracking_torch.start()
    process_play_data.start()


    ####################################################################################################
    ##  Process 끄기  ###################################################################################
    ####################################################################################################
    process_play_data.join()

    # A. 비 실시간 테스트 (모든 frame)
    # process_tracking_unreal_all_frames.join()
    # process_gaussian_mapping_unreal_all_frames.join()

    # B. 비 실시간 테스트 (Keyframe selection)
    process_tracking_unreal.join()
    process_gaussian_mapping_unreal.join()

    # C. ByPass 테스트
    # process_tracking_bypass.join()
    # process_gaussian_mapping_bypass.join()

    # process_tracking_torch.join()
    # process_mapping.join()
    # process_gaussian_mapping.join()

