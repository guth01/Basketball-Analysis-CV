from .utils import draw_ellipse

class PlayerTrackDrawer:
    def __init__(self):
        pass

    def draw(self, video_frames,tracks):

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            #Draw Player tracks
            for track_id, player in player_dict.items():
                frame = draw_ellipse(frame, player['bbox'],(0,0,255),track_id)

            output_video_frames.append(frame)
        return output_video_frames 
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
