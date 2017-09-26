#################################################################
### Just an example, there is not much to wrap around VideoIO ###
#################################################################

#import VideoIO, Images, ImageInTerminal
#
# fh = VideoIO.open("videofile")
# f = VideoIO.open(fh)
#
## or 
#
# h=VideoIO.openvideo("/home/zgornel/projects/DATA/video/pontsync.avi")
#
# img = read(h) # reads stuff
#

# Examples from the High-Level interface:
#=
using Images
import ImageView
import VideoIO

io = VideoIO.open(video_file)
f = VideoIO.openvideo(io)

# As a shortcut for just video, you can upen the file directly
# with openvideo
#f = VideoIO.openvideo(video_file)

# Alternatively, you can open the camera with opencamera().
# The default device is "0" on Windows, "/dev/video0" on Linux,
# and "Integrated Camera" on OSX.  If using something other
# than the default, pass it in as the first parameter (as a string).
#f = VideoIO.opencamera()

# One can seek to an arbitrary position in the video
seek(f,2.5)  ## The second parameter is the time in seconds
img = read(f, Image)
canvas, _ = ImageView.view(img)

while !eof(f)
	read!(f, img)
	ImageView.view(canvas, img)
	#sleep(1/30)
end

=#

