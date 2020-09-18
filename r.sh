export LD_LIBRARY_PATH=/home/wzw/ffmpeg_build/lib:$LD_LIBRARY_PATH
export PATH=/home/wzw/bin:$PATH
ffmpeg -i 480p.jpg -vf format=yuv420p,scale=w=iw*2:h=ih*2,dnn_processing_async=dnn_backend=openvino:model=srcnn.xml:input=x:output=srcnn/Maximum -y srcnn.ov.async.jpg
ffmpeg -i 480p.mp4 -vf format=yuv420p,scale=w=iw*2:h=ih*2,dnn_processing_async=dnn_backend=openvino:model=srcnn.xml:input=x:output=srcnn/Maximum -y srcnn.ov.async.mp4
ffmpeg -i 480p.jpg -vf format=yuv420p,scale=w=iw*2:h=ih*2,dnn_processing_async=async=0:dnn_backend=openvino:model=srcnn.xml:input=x:output=srcnn/Maximum -y srcnn.ov.sync.jpg
ffmpeg -i 480p.mp4 -vf format=yuv420p,scale=w=iw*2:h=ih*2,dnn_processing_async=async=0:dnn_backend=openvino:model=srcnn.xml:input=x:output=srcnn/Maximum -y srcnn.ov.sync.mp4
