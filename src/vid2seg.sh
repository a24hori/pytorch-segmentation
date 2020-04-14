# Generate segmented images from input video
echo "Convert video to images..." 
INPUT_VIDEO_PATH=kyoto-city.mp4
SEP_IMG_PATH=../data/kyoto/kyoto-city-images/eval/
IMG_NAME=kyoto-city_%06d.png
echo "video: "$INPUT_VIDEO_PATH
echo "path of converted images: "$SEP_IMG_PATH
mkdir -p $SEP_IMG_PATH
ffmpeg -i $INPUT_VIDEO_PATH -vcodec png $SEP_IMG_PATH$IMG_NAME
echo -n "Number of Images Created: "
ls -1 $SEP_IMG_PATH | wc -l

echo "Generate semantic segmented images..."
python generate.py ../config/eval/cityscapes_deeplab_v3_plus.yaml
