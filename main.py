from utils.video_utils import read_video,save_video
from tracker import PlayerTracker,BallTracker
from drawers import(
    PlayerTrackDrawer,
    BallTracksDrawer
)
def main():
    print("Welcome to the Basketball Project!")
    video_frames = read_video("input_videos/video_1.mp4")
    player_tracker = PlayerTracker("models/player.pt")
    ball_tracker = BallTracker("models/ball.pt")
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracks_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path = "stubs/ball_track_stubs.pkl")

    # Remove wrong ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    #Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    #Draw Output
    player_tracks_drawer = PlayerTrackDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames,ball_tracks)
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()