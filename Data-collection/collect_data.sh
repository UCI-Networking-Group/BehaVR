#!/bin/bash

# Function to print usage
print_usage()
{
	echo ""
	echo "A utility script to start data collection."
	echo -e  "Usage:\tcollect_data.sh <logcat-file-name>"
	echo ""
	echo -e "\t-h\t(print this usage info)"
	echo ""
	echo -e "\t-f\t<logcat-file-name>"
	echo ""
	exit 1
}

###
# Main body of script
###
# Get input argument and execute the right function
if [[ $1 == '-f' ]]
then
	# Clean up logcat output and save the new stream into a specified file
	adb logcat -c; adb logcat > $2
else
	# Print usage info if there is any mistake
	print_usage
fi
