#!/bin/bash
# src_dir="data/200413/interactive_proposal_divide_training_03_propose_04/visualize"
# target_dir="data/200413/interactive_proposal_divide_training_03_propose_04/gifs"

src_dir="data/200927/universal_joint_init1v1_2000_old_eval_risk/recording"
target_dir="data/200927/universal_joint_init1v1_2000_old_eval_risk/gifs"


mkdir -p "$target_dir"

for file in `ls "$src_dir/"*.mp4`;
do
    filename=`basename $file .mp4`
    echo "$filename"
    ffmpeg -i "$src_dir"/"$filename".mp4 -r 10 "$target_dir/$filename".gif
done


ffmpeg -i "$filename".mp4 -r 10 "$filename".gif
