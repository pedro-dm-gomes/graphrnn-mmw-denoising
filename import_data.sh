#!/bin/bash

# set src_dir variable to a string
src_dir="/home/walter/turtlebot3_ws/src/turtle/script/data/run/"

# list all directories from run58 to run85
list=$(ls $src_dir | grep run[5-9][0-9] | grep -v run5[0-7])

# for loop to print $src_dir$element of a list
for element in $list
do
    src_id=${element: -2}
    dest='data/run_'$(($src_id+13))
    # echo ${element:0:3}$temp
    echo 'Elaborating dir: '$dest
    mkdir $dest
    # cp $src_dir'run'$src_id'/data_run'$src_id $dest'/data_run'$(($src_id+13))
    cp $src_dir'run'$src_id'/dataset_run'$src_id'.json' $dest'/dataset_run'$(($src_id+13))'.json'
    cp $src_dir'run'$src_id'/labelled_mmw_run'$src_id'.json' $dest'/labelled_mmw_run'$(($src_id+13))'.json'
    cp $src_dir'run'$src_id'/norm_mmw_run'$src_id'.json' $dest'/norm_mmw_run'$(($src_id+13))'.json'
done

# echo $src_dir$element 'data/'${element:0:3}'_'$temp
# cp -r $src_dir$element 'data/'${element:0:3}'_'$temp