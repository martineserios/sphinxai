#!/bin/sh
for entry in "../media_tmp"/*
do
  echo $entry
  python video_proc_script.py -a MarinaBulbarella -v $entry
done