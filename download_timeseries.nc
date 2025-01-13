#!/bin/bash 
source ../environments/EE2/bin/activate
stepsize=2

for n in {0..1}; 
do
    echo $(($n*$stepsize))
    echo $(($(($n*$stepsize))+$stepsize))
    python EE_download_timeseries.py -s satdata$(($n*$stepsize)) -start $(($n*$stepsize)) -end $(($(($n*$stepsize))+$stepsize))
done
