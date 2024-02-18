from engine import Engine

# Test the Engine class
video_paths = ["videos/cia.mp4"]    # add your video file paths here
engine = Engine(video_paths)
engine.extract_metadata()
engine.save_metadata("metadatas.json")
print("Metadata saved to metadata.json")