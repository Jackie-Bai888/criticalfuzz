#!/bin/bash
while true
do
        echo $`ps -p 3818183 -o etime` >> run_time.txt
	sleep 60
done
