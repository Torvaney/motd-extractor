from multiprocessing import Pool

import moviepy.editor as mpy
import numpy as np
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_peaks


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


def find_clip_corners(clip, sample_times, n_workers=4):
    """
    Find the number of corners in a selection of a moviepy clip's frames in parallel
    :param clip: moviepy video clip
    :param sample_times: iterable of the times (in seconds) of the frames to be used
    :param n_workers: number of workers to be used by multiprocessing.Pool
    :return: a list containing the number of corners in at each of the times in sample_times
    """

    # multiprocessing.Pool requires a named function with a single argument
    def find_frame_corners(frame_time):
        return find_image_corners(clip.get_frame(frame_time))

    # Find number of corners in each frame in parallel
    workers = Pool(n_workers)
    n_corners = workers.map(find_frame_corners, sample_times)

    return n_corners


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'same')
    return sma


def get_highlight_times(corners_list, sampling_rate, smoothing_window=30):
    """
    Get the start and stop times of highlights footage based on the number of corner peaks throughout the clip.
    :param corners_list: A list of the number of corners at frames taken from the clip at the given sampling rate
    :param sampling_rate: Sampling rate of the clip that the corners were calculated for
    :param smoothing_window: Width of window (in seconds) for the rolling average to be calculated
    :return: list of start and stop times for highlights.
    """

    rolling_corners = moving_average(corners_list, smoothing_window * sampling_rate)
    is_highlights = np.where([rolling_corners > np.mean(rolling_corners)], 1, 0)[0]

    changes = np.diff(is_highlights)
    start_times = np.where(changes == 1)[0] / sampling_rate
    stop_times = np.where(changes == -1)[0] / sampling_rate

    return start_times, stop_times


def extract_highlights(
        clip, file_name='output.mp4',
        xlim=(0.085, 0.284), ylim=(0.05, 0.1),
        sampling_rate=1, minimum_clip=60,
        buffer_length=(7, 7), parallel_workers=4):
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
    :param parallel_workers: Number of parallel workers to be passes to multiprocessing.Pool
    :return: None
    """

    width, height = get_resolution(clip)

    # Crop to where the scoreboard is during highlights (defaults to top left corner)
    box_clip = clip.crop(x1=xlim[0] * width, x2=xlim[1] * width,
                         y1=ylim[0] * height, y2=ylim[1] * height)

    # Get number of 'corners' for each frame at given sampling rate
    frame_times = np.arange(0, box_clip.duration, 1 / sampling_rate)
    n_corners = find_clip_corners(box_clip, frame_times, parallel_workers)

    start_times, stop_times = get_highlight_times(n_corners, sampling_rate)

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
