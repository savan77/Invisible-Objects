## Invisible Objects

Make any objects invisible using Deep Learning and Image Processing.
You can make any of the 60 objects (see labels.txt) invisible simultaneously. This can also be extended to other objects if the model can segment that object.

### Execution

```
python main.py --video=sample.mp4 --out_video=sample_output.mp4 --show=True
```

The `show` argument suggests that we want to see output frames during the process. While running the script, you need to hit enter to proceed to next frame.
If you want to make any objects invisible at a moment, enter the class id of that object (i.e 15 for person).

If you want make invisible objects visible, just enter class id preceeded by c (i.e c15 for person).

## Change Background

You can also leverage this script to change background. You can keep objects of interest in foreground.
![demo](./assets/thumb.png)

Below command shows how to run this.

```
python background_change.py --video=./assets/bgdemo3.mp4 --out_video=sample.mp4 --show=True --bg=./assets/bg_beach.jpg --change_bg_dynamic=True
```