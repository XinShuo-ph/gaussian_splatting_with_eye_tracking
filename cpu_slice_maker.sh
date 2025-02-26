# make a systemd slice
# takes 1 argument: slice name
# runs as sudo ./cpu_slice_maker.sh slice_name

# create slice
echo "[Slice]
CPUQuota=100%" > /etc/systemd/system/$1.slice

systemctl daemon-reload

#  在隔离环境中运行脚本
# systemd-run --slice=cpu-2.2ghz.slice python3 /home/ubuntu/test.py


# to monitor the CPU usage by slice: 
# systemd-cgtop