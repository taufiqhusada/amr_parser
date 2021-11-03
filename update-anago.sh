#!/bin/bash

github_url="https://raw.githubusercontent.com/banditelol/anago/"
while getopts d: flag
do
    case "${flag}" in
        d) directory=${OPTARG};;
        c) commit_hash=${OPTARG};;
    esac
done
if [ -z ${directory+x} ]; then
    echo "directory is unset, using venv directory";
    directory="venv/lib/py*/site-packages/anago";
fi

if [ -z ${commit_hash+x} ]; then
    echo "commit_hash is unset, using default hash directory";
    commit_hash="9afccaa5bcc232676f9c2b59faa4c9531fb25190";
fi

echo "anago directory: '$directory'";
echo "commit hash: '$commit_hash'";
echo "check $github_url$commit_hash for included files"
filenames=(\
"callbacks.py" \
"layers.py" \
"models.py" \
"preprocessing.py" \
"utils.py")
for fn in ${filenames[@]}; do
    echo "updating $fn"
    wget -nv $github_url$commit_hash/anago/$fn -O $directory/$fn;
done
