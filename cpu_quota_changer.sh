# alternate between 0.5 and 1.0 CPU quota using sudo echo "50000\ 100000" > /sys/fs/cgroup/_cpu.slice/cpu-2.2ghz.slice/cpu.max
# takes 1 argument: the time lapse between each change in seconds
# example: sudo ./cpu_quota_changer.sh 0.5
# note that this script must be run by root, even if the user is in the sudoers list

# check if the argument is provided
if [ -z "$1" ]; then
    echo "Please provide the time lapse between each change in seconds"
    exit 1
fi

# check if the argument is a number
if ! [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Please provide a number as the argument"
    exit 1
fi

# start the loop
while true; do
    echo "50000 100000" > /sys/fs/cgroup/_cpu.slice/cpu-2.2ghz.slice/cpu.max
    sleep $1
    echo "100000 200000" > /sys/fs/cgroup/_cpu.slice/cpu-2.2ghz.slice/cpu.max
    sleep $1
done