from utils.video_utils import read_video,save_video
from tracker import PlayerTracker
def main():
    print("Welcome to the Basketball Project!")
    video_frames = read_video("input_videos/video_3.mp4")
    player_tracker = PlayerTracker("models/player.pt")
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracks_stubs.pkl")
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()