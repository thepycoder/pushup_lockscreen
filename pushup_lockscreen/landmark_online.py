from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer


class PreprocessorVideo:
    def __init__(self):
        self.detector = BlazeposeDepthai(
            input_src='rgb',
            pd_score_thresh=0.5,
            lm_score_thresh=0.5,
            pd_model=None,
            lm_model='../depthai/models/pose_landmark_heavy_sh4.blob',
            smoothing=True,
            xyz=False,
            crop=False,
            internal_fps=None,
            internal_frame_height=300,
            force_detection=False,
            stats=False,
            trace=False
        )

        self.renderer = BlazeposeRenderer(
            self.detector,
            show_3d=False,
            output=None
        )

    def get_next_frame(self):
        # Run blazepose on next frame
        frame, body = self.detector.next_frame()
        if frame is None:
            print("No next frame found!")
            self.renderer.exit()
            self.detector.exit()
            return None, None
        # Draw 2d skeleton
        frame = self.renderer.draw(frame, body)
        key = self.renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            self.renderer.exit()
            self.detector.exit()
            return None, None

        # Return the frame and the landmarks
        return frame, body
