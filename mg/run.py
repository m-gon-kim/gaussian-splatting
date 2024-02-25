
from tum_dataset import TumDataset
# from tracking import Tracker
from tracking_torch import TrackerTorch
from mapping import Mapper
from mtf_mapping import MTFMapper
from gaussian_mapping import GaussianMapper
import torch.multiprocessing as mp

def PlayTumDataset(img_pair_q):
    dataset = TumDataset()
    begin_index = 1
    cnt = 2400
    awake = True
    for index in range(cnt):
        rgb, gray, d = dataset.ReturnData(index + begin_index)
        # rgb, gray, d = dataset.ReturnData(cnt - index)
        img_pair_q.put([awake, [rgb, gray, d]])
    # img_pair_q.put([False])
    # return

# def TrackingTest(img_pair_q, tracking_result_q,):
#     tracker = Tracker()
#     awake = True
#     while True:
#         if not img_pair_q.empty():
#             instance = img_pair_q.get()
#             if not instance[0]:  # Abort (System is not awake)
#                 print("Tracking Abort")
#                 awake = False
#                 tracking_result_q.put([awake, []])
#                 return
#             tracking_result = tracker.Track(instance)
#             if tracking_result[0][0]:  # Mapping is required
#                 tracking_result_q.put([awake, tracking_result])
#                 print(f"Tracking result: {tracking_result_q.qsize()}")

def TrackingTorch(img_pair_q, tracking_result_q,):
    tracker = TrackerTorch()
    frame = 0

    awake = True
    while True:
        if not img_pair_q.empty():
            frame+=1
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

#
# def MappingTest(tracking_result_q, mapping_result_q):
#     mapper = Mapper()
#     cntr = 0
#     while True:
#         if not tracking_result_q.empty():
#             q_size= tracking_result_q.qsize()
#             print(f"PROCESS: MAPPING Q {q_size}")
#             instance = tracking_result_q.get()
#             if not instance[0]:  # Abort (System is not awake)
#                 mapper.DetectLoop()
#                 print("Mapping Abort")
#                 # mapper.FullBundleAdjustment()
#                 mapping_result_q.put([instance[0], []])
#                 return
#             mapping_result = mapper.Map(instance)
#             if mapping_result[0][0]:
#                 cntr += 1
#                 mapping_result_q.put([True, mapping_result])
#             # if cntr%100 == 30:
#             #     cntr =0
#             #     ba_result = mapper.FullBundleAdjustment(10)
#             #     if ba_result[0][2]:
#             #         mapping_result_q.put([True, ba_result])
#         else:
#             ba_result = mapper.FullBundleAdjustment(10)
#             if ba_result[0][2]:
#                 mapping_result_q.put([True, ba_result])
#         # ba_result = mapper.FullBundleAdjustment()
#         # if ba_result[0][2]:


def MTF_Mapping(tracking_result_q, mapping_result_q):
    mapper = MTFMapper()
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
            if mapping_result[0][0]:
                mapping_result_q.put([True, mapping_result])
            if mapping_result[0][4]:
                mapper.Map()
        # else:
        #     mapping_result = mapper.FullBACall()
        #     if mapping_result[0][2]:
        #         mapping_result_q.put([True, mapping_result])


def GaussianMappingTest(mapping_result_q):
    gaussian_mapper = GaussianMapper()
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
    img_pair_q = mp.Queue   ()
    tracking_result_q = mp.Queue()
    mapping_result_q = mp.Queue()

    process_play_data = mp.Process(target=PlayTumDataset, args=(img_pair_q,))
    # process_tracking = mp.Process(target=TrackingTest, args=(img_pair_q, tracking_result_q,))
    process_tracking_torch = mp.Process(target=TrackingTorch, args=(img_pair_q, tracking_result_q,))
    process_mapping = mp.Process(target=MTF_Mapping, args=(tracking_result_q, mapping_result_q,))
    # process_mapping = mp.Process(target=MappingTest, args=(tracking_result_q, mapping_result_q,))
    process_gaussian_mapping = mp.Process(target=GaussianMappingTest, args=(mapping_result_q,))

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

