#!/bin/bash

# Function to print usage
print_usage()
{
	echo ""
	echo "A utility script to extract all sensor data and generate all CSV files from the logcat output."
	echo -e  "Usage:\tgenerate_all_csvs.sh [options] <logcat-file-name>"
	echo ""
	echo -e "\t-h\t(print this usage info)"
	echo ""
	echo -e "\t-f\t<logcat-file-name>"
	echo ""
	exit 1
}




# Get sensor data
generate_all_csvs()
{
	# First argument is the logcat file name
	LOGCAT_FILENAME=$1
	./generate_csv.sh $LOGCAT_FILENAME -t device_motion
	./generate_csv.sh $LOGCAT_FILENAME -t device_motion_0
	./generate_csv.sh $LOGCAT_FILENAME -t device_motion_1
	./generate_csv.sh $LOGCAT_FILENAME -t device_motion_2
	./generate_csv.sh $LOGCAT_FILENAME -t left_hand
	./generate_csv.sh $LOGCAT_FILENAME -t right_hand
	./generate_csv.sh $LOGCAT_FILENAME -t left_eye
	./generate_csv.sh $LOGCAT_FILENAME -t right_eye
	mv $LOGCAT_FILENAME $LOGCAT_FILENAME.log
	mkdir $LOGCAT_FILENAME
	mv $LOGCAT_FILENAME*.csv $LOGCAT_FILENAME.log $LOGCAT_FILENAME
}


###
# Main body of script
###
# Get input argument and execute the right function
if [[ $1 == '-f' ]]
then
	generate_all_csvs $2
else
	# Print usage info if there is any mistake
	print_usage
fi
