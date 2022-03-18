#!/bin/bash

cd $1
mkdir ../../excluded
for xml in *; do 
	if [[ -z `sed -n 19p $xml` ]]; then
		echo $xml
		mv $xml ../../excluded
	fi
done
