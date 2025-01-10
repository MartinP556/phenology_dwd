#!/bin/bash 
source ../environments/EE2/bin/activate
stepsize=2
n=0
python EE_download_timeseries.py -s satdata$n -start $(($n*$stepsize)) -end $(($(($n*$stepsize))+$stepsize))