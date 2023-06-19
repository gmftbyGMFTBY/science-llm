#!/bin/bash

shuf train.txt -o train_shuffle.txt
split -l 166084 train_shuffle.txt -d split_
