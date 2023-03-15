#!/bin/bash

# Function to print usage
print_usage()
{
	echo ""
	echo "A utility script to extract a category of sensor data and generate the CSV file from the logcat output."
	echo -e  "Usage:\generate_csv.sh [options] <logcat-file-name>"
	echo ""
	echo -e "\t-h\t(print this usage info)"
	echo ""
	echo -e "\t<file-name>\t-t\t<data-type>, i.e., device_motion, left_hand, right_hand, left_eye, right_eye"
	echo ""
	exit 1
}




# Get sensor data
generate_csv()
{
	# First argument is the data type
	LOGCAT_FILENAME=$1
	DATA_TYPE=$2
	# Get the data type of interest from the logcat output
	if [[ 	$DATA_TYPE == 'device_motion' 	|| $DATA_TYPE == 'device_motion_0'	|| $DATA_TYPE == 'device_motion_1'	|| $DATA_TYPE == 'device_motion_2' ||
			$DATA_TYPE == 'left_hand'		|| $DATA_TYPE == 'right_hand'		|| 
			$DATA_TYPE == 'left_eye' 		|| $DATA_TYPE == 'right_eye' ]]
	then
		echo "==> Extracting $DATA_TYPE from $LOGCAT_FILENAME..."
		# Choose the template based on data type
		if [[ $DATA_TYPE == "device_motion"* ]]; then
			cp templates/device_motion.csv ./$LOGCAT_FILENAME\_$DATA_TYPE.csv
		elif [[ $DATA_TYPE == *"hand" ]]; then
			cp templates/hand.csv ./$LOGCAT_FILENAME\_$DATA_TYPE.csv
		else # [[ $DATA_TYPE == *"eye" ]]
			cp templates/eye.csv ./$LOGCAT_FILENAME\_$DATA_TYPE.csv
		fi
		grep -r "alvr_send_tracking(): $DATA_TYPE" $LOGCAT_FILENAME > $LOGCAT_FILENAME\_$DATA_TYPE\_tmp
		echo "==> Saving $DATA_TYPE into file..."
		awk '{$1=$2=$3=$4=$5=$6=$7=$8=$9=$10=$11=$12=""; print $0}' $LOGCAT_FILENAME\_$DATA_TYPE\_tmp > $LOGCAT_FILENAME\_$DATA_TYPE\_clean\_tmp
		awk '{$1=$1};1' $LOGCAT_FILENAME\_$DATA_TYPE\_clean\_tmp >> $LOGCAT_FILENAME\_$DATA_TYPE.csv
		rm -rf *tmp*
		echo "==> Done processing!"
	else
		# Print usage info if there is any mistake
		print_usage
	fi
}


###
# Main body of script
###
# Get input argument and execute the right function
if [[ $2 == '-t' ]]
then
	generate_csv $1 $3
else
	# Print usage info if there is any mistake
	print_usage
fi
