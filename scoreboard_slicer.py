import moviepy.editor as mpy
import numpy as np
from progressbar import ProgressBar
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_peaks
from multiprocessing import Pool


def load_video(filename='motd-sample.mp4', in_folder=True):
    video_loc = './video/' if in_folder else ''
    clip = mpy.VideoFileClip(video_loc + filename)
    return clip


def get_resolution(clip):
    """ Get the resolution of a moviepy clip (width, height)"""
    sample_frame = clip.get_frame(0)
    return len(sample_frame[0]), len(sample_frame)


def find_image_corners(image, min_dist=5):
    """
    Get the number of Harris corner peaks in an image.
    :param image: array of RGB values
    :param min_dist: minimum distance for two separate peaks to be identified
    :return: number of corner peaks in the image as int
    """
    bw_image = rgb2grey(image)
    corners = corner_peaks(corner_harris(bw_image), min_distance=min_dist)
    return len(corners)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'same')
    return sma


def extract_highlights(
        clip, file_name='output.mp4',
        xlim=(0.085, 0.284), ylim=(0.05, 0.1),
        sampling_rate=1, minimum_clip=60,
        buffer_length=(7, 7)):
    """
    Extracts highlights from soccer video (primarily Match of the Day) using the presence of a scoreboard
    :param clip: MoviePy VideoClip object contaning the full video to be trimmed.
    :param file_name: Output file name.
    :param xlim: List of 2 floats containing the horizontal parts of the screen to scan for the scoreboard. Numbers must
    be between 0 and 1, where xlim=[0, 1] selects the full width of the screen.
    :param ylim: As xlim but for the vertical section of the screen.
    :param sampling_rate: Rate (in fps) to check video for the precense of a scoreboard.
    :param minimum_clip: Threshold for a  continuous section of video to be included in output in seconds.
    :param buffer_length: Length of buffer in seconds to add before and after each set of hihglights.
    :return: None
    """

    width, height = get_resolution(clip)

    # Crop to where the scoreboard is during highlights (defaults to top left corner)
    box_clip = clip.crop(x1=xlim[0] * width, x2=xlim[1] * width,
                         y1=ylim[0] * height, y2=ylim[1] * height)

    # Get number of 'corners' for each frame at given sampling rate
    frame_times = np.arange(0, box_clip.duration, 1 / sampling_rate)

    # multiprocessing.Pool requires a named function with a single argument
    def find_clip_corners(sample_times):
        return [find_image_corners(box_clip.get_frame(t)) for t in sample_times]

    # Find number of corners in each frame in parallel
    workers = Pool(4)
    n_corners = workers.map(find_clip_corners, frame_times)

    rolling_corners = moving_average(n_corners, 30 * sampling_rate)
    is_highlights = np.where([rolling_corners > np.mean(rolling_corners)], 1, 0)[0]

    changes = np.diff(is_highlights)
    start_times = np.where(changes == 1)[0]
    stop_times = np.where(changes == -1)[0]

    highlight_times = [(start, stop) for start, stop in zip(start_times, stop_times) if (stop - start) >= minimum_clip]

    # get highlights in a list
    highlights = [clip.subclip(t_start=t[0] - buffer_length[0], t_end=t[1] + buffer_length[1]) for t in highlight_times]

    # add fade in/out (half buffer length)
    highlights = [h.fadein(buffer_length[0] / 2) for h in highlights]
    highlights = [h.fadeout(buffer_length[1] / 2) for h in highlights]

    # join videos together into one and write to file
    final_clip = mpy.concatenate_videoclips(highlights, method='compose')
    final_clip.write_videofile('./output/' + file_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract highlights from Match of the Day.')
    parser.add_argument('input', help='Name of input file in \'video\' directory', type=str)
    parser.add_argument('output', help='Name of new output file in \'output\' directory', type=str)

    args = parser.parse_args()

    video_clip = load_video(args.input)
    extract_highlights(video_clip, args.output)
