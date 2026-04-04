from utils.video_utils import read_video,save_video
from tracker import PlayerTracker,BallTracker
from drawers import(
    PlayerTrackDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer
)
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
def main():
    print("Welcome to the Basketball Project!")
    video_frames = read_video("input_videos/video_3.mp4")
    player_tracker = PlayerTracker("models/player.pt")
    ball_tracker = BallTracker("models/ball.pt")
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_tracks_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path = "stubs/ball_track_stubs.pkl")

    # Remove wrong ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    #Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    #Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
    video_frames,
    player_tracks,
    read_from_stub=True,
    stub_path="stubs/player_assignment_stub.pkl"
)
    
    #Ball Aquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    #Detect Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions=pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    #Draw Output
    #Initialize Drawers
    player_tracks_drawer = PlayerTrackDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_assignment,ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames,ball_tracks)

    #Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,player_assignment,ball_aquisition)

    #Draw Passes and Interceptions
    output_video_frames = pass_interception_drawer.draw(output_video_frames,passes,interceptions)
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()