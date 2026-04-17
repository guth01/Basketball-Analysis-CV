from helpers.io import read_video, save_video
from detection import PlayerDetector, BallDetector
from renderers import (
    PlayerRenderer,
    BallRenderer,
    ControlRenderer,
    EventsRenderer,
    ActionRenderer,
    MotionRenderer,
)
from classifier import TeamClassifier
from possession import PossessionDetector
from event_detector import GameEventDetector
from action_classifier import ActionClassifier
from motion_analyzer import MotionAnalyzer
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
    player_detector = PlayerDetector("models/player.pt")
    ball_detector = BallDetector("models/ball_detector_model.pt")
    player_tracks = player_detector.get_object_tracks(video_frames, read_from_stub=True, stub_path=f"stubs/player_tracks_stubs_{video_name}.pkl")
    ball_tracks = ball_detector.get_object_tracks(video_frames, read_from_stub=True, stub_path=f"stubs/ball_track_stubs_{video_name}.pkl")

    # Remove wrong ball detections
    ball_tracks = ball_detector.remove_wrong_detections(ball_tracks)
    # Interpolate ball tracks
    ball_tracks = ball_detector.interpolate_ball_positions(ball_tracks)

    # Assign player teams
    team_classifier = TeamClassifier()
    player_assignment = team_classifier.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=True,
        stub_path=f"stubs/player_assignment_stub_{video_name}.pkl"
    )

    # Ball Possession
    possession_detector = PossessionDetector()
    ball_possession = possession_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Detect Passes and Interceptions
    game_event_detector = GameEventDetector()
    passes = game_event_detector.detect_passes(ball_possession, player_assignment)
    interceptions = game_event_detector.detect_interceptions(ball_possession, player_assignment)

    # ------------------------------------------------------------------ #
    # Action Classification & Jersey Annotation (unified loop)           #
    # ------------------------------------------------------------------ #
    action_classifier = ActionClassifier(fps=30)
    from annotation import NumberAnnotator
    number_annotator = NumberAnnotator()

    shot_events_by_frame = {}
    total_frames = len(video_frames)
    print(f"[Pipeline] Processing {total_frames} frames for Action Classification and Annotation...")

    for frame_index, frame in enumerate(video_frames):
        if frame_index % 100 == 0:
            print(f"  [Pipeline] Frame {frame_index}/{total_frames}")

        # One shared inference call
        detections = action_classifier.run_inference(frame)

        # 1. Feed Action Classifier
        events = action_classifier.update(frame_index, detections)
        if events:
            shot_events_by_frame[frame_index] = events

        # 2. Feed Number Annotator
        player_tracks_for_frame = player_tracks[frame_index]
        number_annotator.update(frame, frame_index, player_tracks_for_frame, detections)

    # Get final jersey labels
    all_track_ids = set()
    for frame_tracks in player_tracks:
        all_track_ids.update(frame_tracks.keys())
    jersey_labels = number_annotator.get_labels(list(all_track_ids))

    action_classifier.save_events(Path(f"output_videos/{output_name}_shot_events.json"))

    # Motion Analysis
    # Using video dimensions and standard FIBA court dimensions (28x15 meters) as naive approximations
    motion_analyzer = MotionAnalyzer(
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

    player_distances_per_frame = motion_analyzer.calculate_distance(tactical_player_positions)
    player_speed_per_frame = motion_analyzer.calculate_speed(player_distances_per_frame)

    # Draw Output — Initialize Renderers
    player_renderer = PlayerRenderer()
    ball_renderer = BallRenderer()
    control_renderer = ControlRenderer()
    events_renderer = EventsRenderer()
    action_renderer = ActionRenderer(fps=30)
    motion_renderer = MotionRenderer()

    output_video_frames = player_renderer.draw(video_frames, player_tracks, player_assignment, ball_possession, jersey_numbers=jersey_labels)
    output_video_frames = ball_renderer.draw(output_video_frames, ball_tracks)

    # Draw Team Ball Control
    output_video_frames = control_renderer.draw(output_video_frames, player_assignment, ball_possession)

    # Draw Passes and Interceptions
    output_video_frames = events_renderer.draw(output_video_frames, passes, interceptions)

    # Draw Action Events (MADE / MISSED banners)
    output_video_frames = action_renderer.draw(output_video_frames, shot_events_by_frame)

    # Draw Motion Overlay
    output_video_frames = motion_renderer.draw(
        output_video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame
    )

    save_video(output_video_frames, f"output_videos/{output_name}.avi")

    # ------------------------------------------------------------------ #
    # Export Analytics Metrics                                            #
    # ------------------------------------------------------------------ #
    print("[Pipeline] Generating Analytics Graphs...")
    # 1. Team Ball Control Pie Chart
    team_ball_control = control_renderer.get_team_ball_control(player_assignment, ball_possession)
    team_1_frames = np.sum(team_ball_control == 1)
    team_2_frames = np.sum(team_ball_control == 2)

    if team_1_frames + team_2_frames > 0:
        plt.figure(figsize=(8, 6))
        plt.pie([team_1_frames, team_2_frames], labels=["Team 1", "Team 2"], autopct='%1.1f%%', startangle=90, colors=['#E0E0E0', '#1E88E5'])
        plt.title("Team Ball Control (Possession %)")
        plt.savefig(f"output_videos/{output_name}_ball_control_pie.png", bbox_inches='tight')
        plt.close()

    # 2. Player Speed Profile Graph (Top 5 fastest players)
    max_speeds = {}
    for frame_speeds in player_speed_per_frame:
        for player_id, speed in frame_speeds.items():
            if player_id not in max_speeds or speed > max_speeds[player_id]:
                max_speeds[player_id] = speed

    # Sort and get top 5 by maximum speed
    top_5_players = sorted(max_speeds.keys(), key=lambda k: max_speeds[k], reverse=True)[:5]

    plt.figure(figsize=(10, 6))
    for player_id in top_5_players:
        player_label = str(player_id)
        if player_id in jersey_labels:
            player_label = f"{player_id} (Jersey {jersey_labels[player_id]})"

        speeds_over_time = [frame_speeds.get(player_id, 0) for frame_speeds in player_speed_per_frame]
        plt.plot(speeds_over_time, label=f"Player {player_label}", linewidth=1.5)

    plt.xlabel("Frame Number")
    plt.ylabel("Speed (km/h)")
    plt.title("Player Speed Profile (Top 5 Active Players)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"output_videos/{output_name}_player_speed_profile.png", bbox_inches='tight')
    plt.close()

    print(f"[Pipeline] Generated {output_name}_ball_control_pie.png and {output_name}_player_speed_profile.png in output_videos/")


if __name__ == "__main__":
    main()
