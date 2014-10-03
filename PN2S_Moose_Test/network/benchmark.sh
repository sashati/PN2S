#!/usr/bin/env bash
set -e
echo "Running PN2S benchmark"
# for s in {1 2 4 8 16 32 50 64 80 100 128 150 200}

# python benchmark.py cpu $model_size 

for stream_number in 8 6 4 2 
do
for model_pack_size in 32 64 128 256 512 800 1024 2048 3072 4086 5120 6144 7168 8192 9216 10240 20480 30720 
do
for model_size in 10 50 80 100 200 400 600 800 1000 1500 2000 4000 8000 10000
do
   	echo "Size = $model_size PackSize = $model_pack_size ST=$stream_number"
	python benchmark.py gpu $model_size $model_pack_size $stream_number
	# | tee restlt.log
done 
done
done