import os
import cv2 
import glob
import re
import numpy as np
import pandas as pd
from .BaseLoader import BaseLoader

class VIPLLoader(BaseLoader):
    def __init__(self, name, data_path, config_data):
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "p*")
        if not data_dirs:
            raise ValueError(self.dataset_name + "data paths empty!")
        
        dirs = [{
            "index": re.search("p(\d+)", data_dir).group(0), "path": data_dir
        } for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        saved_filename = data_dirs[i]['index']
        v_list = [f"v{i}" for i in range(1, 10)]
        source_list = [f"source{i}" for i in range(1, 5)]
        input_name_list_all = list()
        for v in v_list:
            for s in source_list:
                vs_path = os.path.join(data_dirs[i]['path'], v, s)
                if os.path.exists(vs_path):
                    frames = self.read_video(os.path.join(vs_path, "video.avi"))
                    bvps = self.read_wave(os.path.join(vs_path, "wave.csv"))
                    save_name = f"{saved_filename}_{v}_{s}"
                    frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
                    input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, save_name)
                    input_name_list_all = input_name_list_all + input_name_list
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        df = pd.read_csv(bvp_file)
        bvp = df["Wave"].to_numpy()
        return bvp
