

from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset
from scannet_dataset import ScannetDataset

# from tracking import Tracker
from tracking_torch import TrackerTorch
from mapping import Mapper
from mtf_mapping import MTFMapper
from gaussian_mapping import GaussianMapper
import torch.multiprocessing as mp

def GetDataset(dataset_type):
    dataset_dict = {
        "TUM": TumDataset(),
        "REPLICA": ReplicaDataset(),
        "SCANNET": ScannetDataset()
    }
    dataset = dataset_dict.get(dataset_type)
    # dataset.InitializeDataset()
    return dataset

def PlayDataset(dataset, img_pair_q):
    begin_index = 1
    cnt = dataset.get_data_len()
    awake = True

    for index in range(cnt):
        rgb, gray, d = dataset.ReturnData(index + begin_index)
        img_pair_q.put([awake, [rgb, gray, d]])

def TrackingTorch(dataset, img_pair_q, tracking_result_q):
    tracker = TrackerTorch(dataset)

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

def MTF_Mapping(dataset, tracking_result_q, mapping_result_q):
    mapper = MTFMapper(dataset)

    while True:
        if not tracking_result_q.empty():
            q_size= tracking_result_q.qsize()
            print(f"PROCESS: MAPPING Q {q_size}")
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
                loop_close_result = mapper.CloseLoop(mapping_result[4])
                mapping_result_q.put([True, loop_close_result])
                # mapper.PointPtrUpdate()

def GaussianMappingTest(dataset, mapping_result_q):
    gaussian_mapper = GaussianMapper(dataset)

    opt_iter = 0
    viz_iter = 0
    while True:
        if not mapping_result_q.empty():
            q_size = mapping_result_q.qsize()
            print(f"PROCESS: G-MAPPING Q {q_size}")
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

    # EXAMPLE: TUM, REPLICA, SCANNET
    dataset_type = "TUM"
    dataset = GetDataset(dataset_type)

    process_play_data = mp.Process(target=PlayDataset, args=(dataset, img_pair_q,))
    # process_tracking = mp.Process(target=TrackingTest, args=(img_pair_q, tracking_result_q,))
    process_tracking_torch = mp.Process(target=TrackingTorch, args=(dataset, img_pair_q, tracking_result_q,))
    process_mapping = mp.Process(target=MTF_Mapping, args=(dataset, tracking_result_q, mapping_result_q,))
    # process_mapping = mp.Process(target=MappingTest, args=(tracking_result_q, mapping_result_q,))
    process_gaussian_mapping = mp.Process(target=GaussianMappingTest, args=(dataset, mapping_result_q,))

    process_gaussian_mapping.start()
    process_mapping.start()
    process_tracking_torch.start()
    process_play_data.start()
    # process_tracking.start()

    process_play_data.join()
    # process_tracking.join()
    process_tracking_torch.join()
    process_mapping.join()
    process_gaussian_mapping.join()

