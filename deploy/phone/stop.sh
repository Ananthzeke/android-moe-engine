#!/system/bin/sh
# Match the process name: a server started as ./ame-server has a relative command line.
pkill -x ame-server || true
