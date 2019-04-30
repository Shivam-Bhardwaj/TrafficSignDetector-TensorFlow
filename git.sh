#!/bin/bash

if [ -z "${VARIABLE}" ]; then 
    FOO='Random Updates'
else 
    FOO=${VARIABLE}
fi

git add .
git commit -m FOO
git push

