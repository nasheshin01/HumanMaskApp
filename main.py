from ast import arg
import sys
import argparse

from typing import List

from processer import Processer

def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_type', choices=['camera', 'video', 'image'], help='Type of input stream to process. Range: camera, video, image')
    parser.add_argument('--input_path', help='If parameter input-type is video or image, set the path to this video or image')
    parser.add_argument('output_type', choices=['window', 'folder', 'image'], help='Type of output stream. Range: window, file')
    parser.add_argument('--output_path', help='If parameter output-type is file, set the path to save output there')
    parser.add_argument('model_weights', help='Path to weights of mask model')
    parser.add_argument('background_type', choices=['blur', 'image'], help='Type of background of mask')
    parser.add_argument('--background_path', help='If parameter background-type is image or video, set the path to this video or image')

    args = parser.parse_args()

    print(args)

    processer = Processer(args.input_type, args.output_type, args.model_weights, args.background_type,
                          input_path=args.input_path, output_path=args.output_path, background_path=args.background_path)
    processer.run()


if __name__ == "__main__":
    main(sys.argv)