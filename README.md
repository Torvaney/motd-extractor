# Automated MotD highlights extractor
This repo contains a small python (3) module for automatically removing the banal and uninsightful commentary from Match of the day, leaving you with just footage of the actual football you wanted to watch in the first place.

A slightly fuller explanation of the method and motivation behind this project can be found in [this blog post](https://statsandsnakeoil.wordpress.com/2016/08/29/improving-match-of-the-day-with-python/)

## Example

You can either use this in python or directly from the command line.

Python:
```python
import scoreboard_slicer as motd

my_clip = motd.load_video('full_fat_motd.mp4')
motd.extract_highlights(my_clip, 'highlights_only.mp4')
```


Terminal:
```terminal
motd-extractor $ python3 scoreboard_slicer.py [full video filename] [output video filename]
```


This will take the video saved in `/motd-extractor/video/[full video filename]` and save a trimmed version to `/motd-extractor/output/[output video filename]`

The module contains two primary functions:
 
 * `load_video` for loading the video (obviously). This is effectively just a wrapper around `moviepy.editor.VideoFileClip`.
 * `extract_highlights` for identifying, trimming and joining back together the match highlights.
 
 And that's more or less all there is to it.
 
