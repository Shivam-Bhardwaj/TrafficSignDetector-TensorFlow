#!/bin/bash

#if [ -z "${VARIABLE}" ]; then 
#    message='Random Updates'
#else 
#    message=${VARIABLE}
#fi

VAR1="${VAR1:-Minor updates}"

git add .
git commit -m VAR1
git push

