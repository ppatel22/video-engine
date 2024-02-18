import numpy as np
import cv2
import os
import subprocess
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, input_video, target_fps=24):
        """
        input_video: filepath for the video to process
        target_fps (optional): the desired frame rate at which to process the video
        """
        self.input_video = input_video
        self.output_audio = self.extract_audio(input_video)
        self.sample_rate = "44100"
        self.target_fps = target_fps
        self.visual_activity = []
        self.scene_scores = []
        self.scene_cuts = []
        self.focus_vals = []
        self.deadzones = []
        self.lowfocus = []

    def calculate_laplacian_variance(self, image):
        """
        image: a single frame of the video
        Returns the variance of the Laplacian of the image, which is a measure of the sharpness of the image
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)
        return laplacian_variance

    def calculate_mean_pixel_activity(self, frame1, frame2):
        '''
        frame1: a frame of the video
        frame2: the next frame in the video
        Returns the average change in pixel intensity between frame1 and frame2
        '''
        difference = cv2.absdiff(frame1, frame2)
        min_change = np.min(difference)
        difference -= min_change
        average_change = np.mean(difference)
        return average_change

    def calculate_scene_score(self, frame1, frame2, weights=None):
        '''
        frame1: a frame of the video
        frame2: the next frame in the video
        weights (optional): a list of weights for the different components of the scene score
        Returns a score that represents the difference between frame1 and frame2, higher score indicates possible scene cut
        '''
        if weights is None:
            weights = [1, 1, 1, 1]

        hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        delta_hue = np.mean(np.abs(hsv_frame1[:, :, 0] - hsv_frame2[:, :, 0]))
        delta_sat = np.mean(np.abs(hsv_frame1[:, :, 1] - hsv_frame2[:, :, 1]))
        delta_lum = np.mean(np.abs(hsv_frame1[:, :, 2] - hsv_frame2[:, :, 2]))

        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        edges_frame1 = cv2.Canny(gray_frame1, 50, 150)
        edges_frame2 = cv2.Canny(gray_frame2, 50, 150)
        delta_edges = np.mean(np.abs(edges_frame1 - edges_frame2))

        score = (
            weights[0] * delta_hue
            + weights[1] * delta_sat
            + weights[2] * delta_lum
            + weights[3] * delta_edges
        )
        return score

    def process_video(self):
        '''
        Processes the video and saves the results to the corresponding attributes.
        For each frame, calculates the visual activity, focus value, and scene score.
        Using a statistical threshold, identifies and saves the scene cuts and low focus frames.
        Same method can be used to find visual hotspots/deadzones.
        '''
        vid = cv2.VideoCapture(self.input_video)
        ret, prev_frame = vid.read()
        original_fps = vid.get(cv2.CAP_PROP_FPS)
        if original_fps < self.target_fps:
            self.target_fps = original_fps
        frames_to_skip = int(original_fps // self.target_fps) - 1

        for _ in tqdm(
            range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing frames"
        ):
            ret, frame = vid.read()
            if not ret:
                break

            self.visual_activity.append(
                self.calculate_mean_pixel_activity(prev_frame, frame)
            )
            self.focus_vals.append(self.calculate_laplacian_variance(frame))
            self.scene_scores.append(self.calculate_scene_score(prev_frame, frame))
            prev_frame = frame.copy()
            if frames_to_skip > 0:
                for _ in range(frames_to_skip):
                    vid.read()

        vid.release()

        scene_threshold = np.mean(self.scene_scores) + 2 * np.std(self.scene_scores)
        for i, score in enumerate(self.scene_scores):
            if score > scene_threshold:
                self.scene_cuts.append(i)

        blur_threshold = np.mean(self.focus_vals) - 2 * np.std(self.focus_vals)
        for i, focus in enumerate(self.focus_vals):
            if focus < blur_threshold:
                self.lowfocus.append(i)

        return None

    def extract_audio(self, video_path):
        '''
        video_path: the filepath of the video from which to extract the audio
        Returns the filepath of the extracted audio
        '''
        filename = video_path.split("/")[-1].split(".")[0]
        output_audio = "audios/" + filename + ".wav"
        sample_rate = "44100"  # 44.1 kHz
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            sample_rate,
            "-ac",
            "2",
            output_audio,
        ]
        # -i: Specifies the input video file.
        # -vn: Indicates that no video should be included in the output.
        # -acodec pcm_s16le: Sets the audio codec to PCM 16-bit little-endian.
        # -ar 44100: Sets the audio sample rate to 44.1 kHz. You can adjust this value as needed.
        # -ac 2: Sets the number of audio channels to 2 for stereo. Adjust as needed

        if (filename + ".wav") not in os.listdir("audios"):
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                print("Command output:\n", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error executing the command:", e)

        return "audios/" + filename + ".wav"

    def process_audio(self):
        '''
        Processes the audio for deadzones.
        Returns a list of lists, where each sublist contains the start and end frames of a deadzone.
        '''
        y, sr = librosa.load(self.output_audio, sr=None)

        frame_length = int(self.sample_rate) // self.target_fps
        print("Processing audio...")
        energy = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=frame_length, center=True
        )[0]

        threshold = energy.mean() - energy.std()
        dead_zones = []
        window_size = 24

        i = 0
        while i < len(energy):
            if energy[i] < threshold:
                start_frame = i
                while i < len(energy) and energy[i] < threshold:
                    i += 1
                if (i - start_frame) >= window_size:
                    end_frame = i - 1
                    dead_zones.append([start_frame, end_frame])
            else:
                i += 1
        print("Audio processing complete.")
        self.deadzones = dead_zones
        return None
    
    def visualize_metric(self, metric):
        '''
        metric: the metric to visualize
        title: the title of the visualization
        '''
        minutes = len(metric) / (self.target_fps * 60)
        plt.plot(metric)
        plt.title(f"{metric} over {minutes:.2f} minutes")
        plt.xlabel("Time (min)")
        plt.ylabel(metric)
        plt.xticks(np.arange(0, minutes[-1], step=1))
        plt.show()
        return None