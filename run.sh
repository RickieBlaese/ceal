# ffmpeg -y -i in.wav -f f32le -ac 1 in.pcm
./bin/ceal
echo " -- exited --"
ffmpeg -y -f f32le -ar 48k -ac 2 -i out.pcm out.wav
