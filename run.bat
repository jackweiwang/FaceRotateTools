CALL conda activate py36

python video.py --pose --landmarks --stips 200 --video_path video
::chcp 65001
:: --pose输出角度 --landmarks输出关键点 --stips 10 指定跳多少帧处理
pause