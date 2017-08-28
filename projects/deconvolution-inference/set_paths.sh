#!/bin/bash

for pkg in `ls -d */`
do
	export PYTHONPATH=$(pwd)/$pkg:$PYTHONPATH
done