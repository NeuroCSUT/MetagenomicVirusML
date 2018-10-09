#!/usr/bin/env bash


hid=1024
layers=2
drop=0.25
activation='relu'
optimizer='adam'
lr_decay=0.95
cw=0.5

for i in `seq 0 18`;
do

  python2 network_clean.py final_LOEO_cw/LOO_"$i"_2ORF_model"$layers"x"$hid"_"$activation"_"$optimizer"_cw"$cw"_drop"$drop"_decay"$lr_decay" --batch_size 100 --epochs 10 --layers $layers --hidden_nodes $hid --dropout $drop --activation $activation --optimizer $optimizer --lr 0.001 --lr_factor $lr_decay --verbose 2 --binary True --datatype concat --class_weight_power $cw --leave_out $i > final_LOEO_cw/LOO_"$i"_2ORF_log_"$layers"x"$hid"_"$activation"_"$optimizer"_cw"$cw"_drop"$drop"_decay"$lr_decay".txt
  sleep 1
done 

