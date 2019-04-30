#!/bin/bash

#if [ -z "${VARIABLE}" ]; then 
#    message='Random Updates'
#else 
#    message=${VARIABLE}
#fi

exp="${VAR1:-Minor updates}"

git add .
git commit -m exp
git push

