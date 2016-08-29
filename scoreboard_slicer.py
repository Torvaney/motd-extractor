import moviepy.editor as mpy
import numpy as np
import progressbar
from skimage.color import rgb2grey
from skimage.feature import corner_harris, corner_peaks


# Load video
def load_video(filename='motd-sample.mp4', in_folder=True):
    video_loc = './video/' if in_folder else ''
    clip = mpy.VideoFileClip(video_loc + filename)
    # change resolution to standardise incoming video & speed up image processing
    return clip


# Get clip's resolution (pixels)
def get_resolution(clip):
    sample_frame = clip.get_frame(0)
    return len(sample_frame[0]), len(sample_frame)


# Take a frame of a movie and return number of corners
def find_corners(image, min_dist=5):
    bw_image = rgb2grey(image)
    corners = corner_peaks(corner_harris(bw_image), min_distance=min_dist)
    return len(corners)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'same')
    return sma


def extract_highlights(clip, file_name='output.mp4',
                       xlim=None, ylim=None,
                       sampling_rate=1, minimum_clip=60,
                       buffer_length=(5, 5)):
    """
    Extracts highlights from soccer video (primarily Match of the Day) using the presence of a scoreboard
    :param clip: MoviePy VideoClip object contaning the full video to be trimmed.
    :param file_name: Output file name.
    :param xlim: List of 2 floats containing the horizontal parts of the screen to scan for the scoreboard. Numbers must
    be between 0 and 1, where xlim=[0, 1] selects the full width of the screen.
    :param ylim: As xlim but for the vertical section of the screen.
    :param sampling_rate: Rate (in fps) to check video for the precense of a scoreboard.
    :param minimum_clip: Threshold for a section of video to be included in output in seconds.
    :param buffer_length: Length of buffer in seconds to add before and after each set of hihglights.
    :return: None
    """

    # Set scoreboard location
    xlim = [0.085, 0.284] if xlim is None else xlim
    ylim = [0.05, 0.1] if ylim is None else ylim

    resolution = get_resolution(clip)

    # Crop to top left corner
    box_clip = clip.crop(x1=xlim[0] * resolution[0], x2=xlim[1] * resolution[0],
                         y1=ylim[0] * resolution[1], y2=ylim[1] * resolution[1])

    # Get number of 'corners' for each frame at given sampling rate
    frame_times = np.arange(0, box_clip.duration, 1 / sampling_rate)
    bar = progressbar.ProgressBar()
    n_corners = [find_corners(box_clip.get_frame(t)) for t in bar(frame_times)]

    rolling_corners = moving_average(n_corners, 30 * sampling_rate)
    is_highlights = np.where([rolling_corners > np.mean(rolling_corners)], 1, 0)[0]

    changes = np.diff(is_highlights)
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]

    start_stop = [(starts[i], stops[i]) for i in range(len(starts)) if (stops[i] - starts[i]) >= minimum_clip]

    # get highlights in a list
    highlights = [clip.subclip(t_start=t[0] - buffer_length[0], t_end=t[1] + buffer_length[1]) for t in start_stop]
    # add fade in/out (half buffer length?)
    highlights = [h.fadein(buffer_length[0] / 2) for h in highlights]
    highlights = [h.fadeout(buffer_length[1] / 2) for h in highlights]

    # join videos together into one and write to file
    final_clip = mpy.concatenate_videoclips(highlights, method='compose')
    final_clip.write_videofile('./output/' + file_name)


