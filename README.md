# Automated MotD highlights extractor
This repo contains a small python (3) module for automatically removing the banal and uninsightful commentary from Match of the day, leaving you with just footage of the actual football you wanted to watch in the first place.


## Example

```python
import scoreboard_slicer as motd

my_clip = motd.load_video('full_fat_motd.mp4')
motd.extract_highlights(my_clip, 'highlights_only.mp4')
```

The module contains two primary functions:
 
 * `load_video` for loading the video (obviously). This is effectively just a wrapper around `moviepy.editor.VideoFileClip`.
 * `extract_highlights` for identifying, trimming and joining back together the match highlights.
 
 And that's more or less all there is to it.
 
