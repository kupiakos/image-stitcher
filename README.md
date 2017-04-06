# image-stitcher
An panoramic image stitching tool made with OpenCV and Python.
It supports arbitrarily large numbers of files, automatically detects centered images,
has good configuration options, and basic color correction!
Transparency in images is also supported by default.

## Example

Say we have these three images in the `demo` directory.

<p>
<img src="/demo/squaw_peak1_600.jpg" width=290 alt="Left Image"/>
<img src="/demo/squaw_peak2_600.jpg" width=290 alt="Center Image"/>
<img src="/demo/squaw_peak3_600.jpg" width=290 alt="Right Image"/>
</p>

We can use the following command to combine the three into one image, performing color correction and giving basic log output.

    ./stitcher.py demo/squaw_peak?_600.jpg -vco demo/squaw_peak_merged_600.png

That results in this image:

<img src="/demo/squaw_peak_merged_600.png"/>

Neat, huh?
For extra documentation, run `./stitcher.py --help`, or just read the source.

## Installation
NumPy, SciPy, Matplotlib, and OpenCV all need to be installed to Python 3.

Most importantly, this script requires an OpenCV build with [SIFT][0] support.
I might implement [ORB][2] support at sometime in the future.
Since SIFT's not installed by default (as it's nonfree), you'll probably have to compile OpenCV on your own.
I highly recommend [this][1] tutorial on building OpenCV with full support for everything.
Just modify it a bit for OpenCV 3.2.



## Bugs
Probably.

[0]: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf "Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe"
[1]: http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/ "Ubuntu 16.04: How to install OpenCV - PyImageSearch"
[2]: http://www.willowgarage.com/sites/default/files/orb_final.pdf "ORB: an efficient alternative to SIFT or SURF"
