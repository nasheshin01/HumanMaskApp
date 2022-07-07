from argparse import ArgumentError
from time import sleep, time
from streamers.input import CameraVideoInputStreamer, FileVideoInputStreamer, ImageInputStreamer
from streamers.output import FolderOutputStreamer, ImageOutputStreamer, WindowOutputStreamer
from transformers import BlurBackgroundTransformer, ImageBackgroundTransformer
from maskers import HumanMaskFinder

class Processer:

    def __init__(self, input_type: str, output_type: str, weights_path: str, background_type:str,
                 input_path: str=None, output_path: str=None, background_path: str=None) -> None:
        self.input_type = input_type
        if input_type == "camera":
            self.input_streamer = CameraVideoInputStreamer()
        elif input_type == "video":
            self.input_streamer = FileVideoInputStreamer(input_path)
        elif input_type == "image":
            self.input_streamer = ImageInputStreamer(input_path)
        else:
            raise ArgumentError()
        
        self.mask_finder = HumanMaskFinder(weights_path)

        if background_type == "blur":
            self.image_transformer = BlurBackgroundTransformer()
        elif background_type == "image":
            self.image_transformer = ImageBackgroundTransformer(background_path)
        else:
            raise ArgumentError()

        if output_type == "window":
            self.output_streamer = WindowOutputStreamer("HumanMaskApp")
        elif output_type == "folder":
            self.output_streamer = FolderOutputStreamer(output_path)
        elif output_type == "image":
            self.output_streamer = ImageOutputStreamer(output_path)
        else:
            raise ArgumentError()

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
            elif isinstance(self.output_streamer, FolderOutputStreamer):
                self.output_streamer.write_frame(transformed_image)
            elif isinstance(self.output_streamer, ImageOutputStreamer):
                self.output_streamer.write_frame(transformed_image)
                break
            else:
                raise ArgumentError()