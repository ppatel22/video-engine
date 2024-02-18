# Introduction

This repository provides an Engine class that can be used to extract the following metadata from a list of one of more videos:
- visual activity
- blurry frames
- scene cuts
- audio deadzones

# Usage

To use this engine, you must first clone the repository and install the required dependencies:

```
pip install -r requirements.txt
```

Also install ffmpeg (assumes you already have Homebrew installed):
```
brew install ffmpeg
```

Create a folder named `videos` to store all videos to process. Create a folder named `audios` to store the extracted audio files.

To run the demo, navigate to `demo.py` and change the list of input video paths, `video_paths`, to include the relative filepaths of every video you would like to process. Please ensure that these videos have been saved to the `videos` folder, and their names do not contain spaces. Run the demo via terminal with `python demo.py` or press "Run" in VSCode. Extracted metadata for all of the videos will be saved to `metadatas.json`.

# Performance
When running the entire pipeline on the "CIA Drug Trafficking Hearings" video, which is 2:36:35 in length, the runtime of my code is 12:52. This gives me a real-time factor of `772 s / 9395 s = 0.0768 << 1`. My device has the M1 Pro chip with 16 GB of ram. Note that this performance will be slightly decreased once approriate post-processing steps are added, as discussed in the "Improvements" section below.

# Methodology
To determine the visual activity per frame, I am using the difference in pixel values between adjacent frames. I figured this is the simplest and most rudimentary way to calculate the enrgy of a video signal. Note that I am offseting the pixel activity by the lowest delta, as this could account for shifts in brightness for the entire frame. For example, a still scene where the sun is setting should have low activity, but pixel values will be changing by approximately the same amount everywhere.

To determine blurry scenes, I calculated the Laplacian variance for each frame. Frames with low Laplacian variance are associated with out-of-focus scenes. This is based on the StackOverflow link provided.

To determine scene cuts, I dug through the source code for the scenedetect library, specifically it's ContentDetector class. There, I found a simple formula to calculate a scalar that is directly proportional to the probability that a frame is a scene cut. This formula looks at the change in hue, saturation, luminosity, and edge count. I reimplemented this code instead of calling the library naively so I could avoid a second pass through the video, which provides a performance boost.

To determine audio deadzones, I am calculating the signal energy for each second of the input video using the librosa library. Then, I find windows where the energy is less than two standard deviations below the mean for the entire video.

# Improvements
This implementation of the metadata engine is far from perfect. For each feature in the metadata, post-processing is needed to remove noise. First, changes in pixel values is a simple way to measure activity, but it does not take into account edge-cases like scene cuts, which should not be labelled as high visual activity. To solve this problem, I would cross reference the frames deemed as high activity with the scene cuts calculated. Second, the threshold values for blurry scenes, scene cuts, and audio deadzones are very arbitraily chosen for now. Given labelled data, it would be beneficial to calculate the ideal threshold values (or a formula that works with all videos). Similarly, weights for the scene cut calculation can be learned, as right now the delta hue, saturation, luminosity, and edge count is being weighed equally. Additionally, my metadata output is not super user-friendly. It is saved as a dictinoary of dictionaaries, without units, and events are associated with frames. The most helpful change  would be to report features with corresponding timestamps, not frame numbers. Lastly, my engine currently assumes perfect inputs: a list of colored videos in .mp4 format with audio. However, it should be able to handle unideal inputs and behave properly when invalid inputs are passed in. Given more time, I would try to add pre and post processing steps that make each of the feature extractions more accurate since runtime is not an issue right now. 

