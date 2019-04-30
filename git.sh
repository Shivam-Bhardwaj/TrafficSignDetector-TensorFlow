#!/bin/bash

if [ -z "${VARIABLE}" ]; then 
    message='Random Updates'
else 
    message=${VARIABLE}
fi

git add .
git commit -m FOO
git push

