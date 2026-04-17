from copy import deepcopy


class GameEventDetector:
    """
    Detects passes between teammates and interceptions by opposing teams.
    """

    def __init__(self):
        pass

    def detect_passes(self, ball_possession, player_assignment):
        """
        Detect successful passes between players of the same team.
        """
        passes = [-1] * len(ball_possession)
        prev_holder = -1
        previous_frame = -1

        for frame in range(1, len(ball_possession)):
            if ball_possession[frame - 1] != -1:
                prev_holder = ball_possession[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_possession[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes

    def detect_interceptions(self, ball_possession, player_assignment):
        """
        Detect interceptions where ball possession changes between opposing teams.
        """
        interceptions = [-1] * len(ball_possession)
        prev_holder = -1
        previous_frame = -1

        for frame in range(1, len(ball_possession)):
            if ball_possession[frame - 1] != -1:
                prev_holder = ball_possession[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_possession[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions
