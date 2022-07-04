from streamers.input import CameraVideoInputStreamer
from streamers.output import WindowOutputStreamer
from transformers import BlurBackgroundTransformer, ImageBackgroundTransformer
from maskers import HumanMaskFinder

class Processer:

    def __init__(self, input_type: str, output_type: str, weights_path: str, background_type:str,
                 input_path: str=None, output_path: str=None, background_path: str=None) -> None:
        self.input_type = input_type
        if input_type == "camera":
            self.input_streamer = CameraVideoInputStreamer()
        
        self.mask_finder = HumanMaskFinder(weights_path)

        if background_type == "blur":
            self.image_transformer = BlurBackgroundTransformer()
        elif background_type == "image":
            self.image_transformer = ImageBackgroundTransformer(background_path)

        if output_type == "window":
            self.output_streamer = WindowOutputStreamer("BlurredBackground")

    def run(self) -> None:
        while(True):
            frame = self.input_streamer.read_frame()
            if frame is None:
                print("Cannot read new frame. Program is going to stop.")
                break
            
            frame_mask = self.mask_finder.get_mask(frame)
            transformed_image = self.image_transformer.transform(frame, frame_mask)

            if isinstance(self.output_streamer, WindowOutputStreamer):
                self.output_streamer.update_window(transformed_image)
                is_program_end = self.output_streamer.check_output_end()
                if is_program_end:
                    break

        

class ProcesserOptions:
    def __init__(self, input_type: str, output_type: str, input_path=None, output_path=None) -> None:
        self.inpu