from utils.video_utils import read_video,save_video
from tracker import PlayerTracker,BallTracker
from drawers import(
    PlayerTrackDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer,
    ShotEventDrawer,
    SpeedAndDistanceDrawer
)
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from shot_detector import ShotDetector
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Basketball Video Analysis")
    parser.add_argument("video_path", type=str, nargs='?', default="input_videos/video_2.mp4", help="Path to input video")
    args = parser.parse_args()

    input_video_path = args.video_path
    video_name = Path(input_video_path).stem
    if video_name.startswith("input_"):
        output_name = video_name.replace("input_", "output_", 1)
    else:
        output_name = f"output_{video_name}"

    print(f"Welcome to the Basketball Project! Processing: {input_video_path}")
    video_frames = read_video(input_video_path)
    player_tracker = PlayerTracker("models/player.pt")
    ball_tracker = BallTracker("models/ball.pt")
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=f"stubs/player_tracks_stubs_{video_name}.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path = f"stubs/ball_track_stubs_{video_name}.pkl")

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
    stub_path=f"stubs/player_assignment_stub_{video_name}.pkl"
)
    
    #Ball Aquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    #Detect Passes and Interceptions
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions=pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # ------------------------------------------------------------------ #
    # Shot Detection (runs in parallel with the existing pipeline)        #
    # ------------------------------------------------------------------ #
    shot_detector = ShotDetector(fps=30)
    shot_events_by_frame = shot_detector.process_video_frames(video_frames)
    shot_detector.save_events(Path(f"output_videos/{output_name}_shot_events.json"))

     # Speed and Distance Calculator
    # Using video dimensions and standard FIBA court dimensions (28x15 meters) as naive approximations
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        width_in_pixels=video_frames[0].shape[1],
        height_in_pixels=video_frames[0].shape[0],
        width_in_meters=28.0,
        height_in_meters=15.0
    )
    
    # Extract player base positions from tracks
    tactical_player_positions = []
    for frame_tracks in player_tracks:
        frame_positions = {}
        for player_id, track_info in frame_tracks.items():
            bbox = track_info['bbox']
            frame_positions[player_id] = ((bbox[0] + bbox[2]) / 2, bbox[3])
        tactical_player_positions.append(frame_positions)

    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)
    #Draw Output
    #Initialize Drawers
    player_tracks_drawer = PlayerTrackDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    shot_event_drawer = ShotEventDrawer(fps=30)
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks,player_assignment,ball_aquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames,ball_tracks)

    #Draw Team Ball Control
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,player_assignment,ball_aquisition)

    #Draw Passes and Interceptions
    output_video_frames = pass_interception_drawer.draw(output_video_frames,passes,interceptions)

    # Draw Shot Events (MADE / MISSED banners)
    output_video_frames = shot_event_drawer.draw(output_video_frames, shot_events_by_frame)

    # Draw Speed and Distance
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames, 
        player_tracks, 
        player_distances_per_frame, 
        player_speed_per_frame
    )

    save_video(output_video_frames, f"output_videos/{output_name}.avi")


if __name__ == "__main__":
    main()
