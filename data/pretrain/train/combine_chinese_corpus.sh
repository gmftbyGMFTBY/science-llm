#!/bin/bash

cat ../chinese_process/chinese_train.json ../english_process/english_train.txt > train.txt
wc -l train.txt
