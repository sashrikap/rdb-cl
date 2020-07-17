#!/bin/bash
# src_dir="data/200413/interactive_proposal_divide_training_03_propose_04/visualize"
# target_dir="data/200413/interactive_proposal_divide_training_03_propose_04/gifs"

src_dir="data/200501/interactive_proposal_divide_training_03_propose_04/mp4"
target_dir="data/200501/interactive_proposal_divide_training_03_propose_04/gifs"


mkdir -p "$target_dir"

for file in `ls "$src_dir/"*.mp4`;
do
    filename=`basename $file .mp4`
    echo "$filename"
    ffmpeg -i "$src_dir"/"$filename".mp4 -r 10 "$target_dir/$filename".gif
done


ffmpeg -i "$filename".mp4 -r 10 "$filename".gif
