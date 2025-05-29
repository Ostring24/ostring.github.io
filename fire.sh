#!/bin/bash

if [ "$1" = "-s" ]; then
  echo "Starting Hugo server in preview mode..."
  hugo server -D
else
  $1

  rm -rf ./resources
  rm -rf ./public
  hugo -D -d docs
fi