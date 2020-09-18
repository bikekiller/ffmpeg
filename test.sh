rm srcnn.ov.async.mp4 srcnn.ov.sync.mp4 srcnn.ov.async.jpg srcnn.ov.sync.jpg

./r.sh
#./r-old.sh

diff srcnn.ov.async.mp4 srcnn.ov.old.mp4
echo "diff srcnn.ov.async.mp4 srcnn.ov.old.mp4 ==> $?"
diff srcnn.ov.async.mp4 srcnn.ov.sync.mp4
echo "diff srcnn.ov.async.mp4 srcnn.ov.sync.mp4 ==> $?"

diff srcnn.ov.async.jpg srcnn.ov.old.jpg
echo "diff srcnn.ov.async.jpg srcnn.ov.old.jpg ==> $?"
diff srcnn.ov.async.jpg srcnn.ov.sync.jpg
echo "diff srcnn.ov.async.jpg srcnn.ov.sync.jpg ==> $?"
