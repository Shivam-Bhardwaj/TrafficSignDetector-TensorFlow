#!/bin/bash

#if [ -z "${VARIABLE}" ]; then 
#    message='Random Updates'
#else 
#    message=${VARIABLE}
#fi

#exp=${exp:-Minor updates}


git add .
git commit -m "$1"
git push

