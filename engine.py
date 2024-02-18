from videoProcessor import VideoProcessor
import json
from tqdm import tqdm

class Engine:
    def __init__(self, input_videos, target_fps=24):
        '''
        input_videos: list of filepaths for videos to process
        target_fps (optional): the desired frame rate at which to process videos
        '''
        self.input_videos = input_videos
        self.video_processors = []
        for input_video in input_videos:
            self.video_processors.append(VideoProcessor(input_video, target_fps))
        self.metadatas = {}

    def extract_metadata(self):
        '''
        Extracts metadata from the videos and saves it to self.metadatas
        '''
        for video in tqdm(self.video_processors, desc="Processing videos"):
            video_path = video.input_video
            filename = video_path.split("/")[-1].split(".")[0]
            video.process_video()
            video.process_audio()

            metadata = {
                "Out of focus frames": video.focus_vals,
                "Scene cuts": video.scene_cuts,
                "Visual activity": video.visual_activity,
                "Audio Dead Zones": video.deadzones,
            }
            self.metadatas[filename]= metadata

        return None
    
    def save_metadata(self, filename):
        '''
        Saves the metadata to a JSON file
        '''
        with open(filename, "w") as file:
            json.dump(self.metadatas, file)
        return None