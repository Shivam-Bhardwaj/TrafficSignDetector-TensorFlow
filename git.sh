#!/bin/bash

if [ -z "${VARIABLE}" ]; then 
    FOO='default'
else 
    FOO=${VARIABLE}
fi

git add .
git commit -m "FOO"
git push

