#!/bin/bash

if [ -z "${VARIABLE}" ]; then 
    'Minor Updates'='default'
else 
    'Minor Updates'=${VARIABLE}
fi

git add .
git commit -m "'Minor Updates'"
git push

