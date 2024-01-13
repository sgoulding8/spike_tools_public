#!/bin/bash

# Define a list of date parameters
dates=(230329 230330 230331)

# Loop through the list of dates
for date in "${dates[@]}"
do
    # Execute the command with the current date parameter
    python spike_tools/mwk_ultra.py --monkey pico --project facescrub-small --date "$date"
done