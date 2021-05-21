#!/usr/bin/env bash
  
set -e
set -u

find /data/ -name "*.pid" -exec sh -c 'ps p `cat {}` > /dev/null 2>&1 && echo {}' \;