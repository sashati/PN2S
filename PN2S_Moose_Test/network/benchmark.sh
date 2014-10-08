#!/usr/bin/env bash
set -e
echo "Running PN2S benchmark"
# for s in {1 2 4 8 16 32 50 64 80 100 128 150 200}

#for stream_number in 1 2 4 8 
for stream_number in 32 
do
for model_pack_size in 32 64 128 256 512 800 1024 2048 3072 4086 8192 9216 10240   
do
   	echo "PackSize = $model_pack_size ST=$stream_number" 1>&2
	python benchmark.py gpu 1000 $model_pack_size $stream_number
#	python benchmark.py gpu 1000 $model_pack_size $stream_number
	# | tee restlt.log
done
done
