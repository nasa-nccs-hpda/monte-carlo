
date_to_delete=$1
echo ps -ef | grep $date_to_delete | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep $date_to_delete | grep -v grep | awk '{print $2}' | xargs -r kill -9
# echo "for pid in $(ps -ef | grep "gtamkin" | grep "Jan21" | awk '{print $2}'); do kill -9  $pid; done"
# "for pid in $(ps -ef | grep "gtamkin" | grep "Jan21" | awk '{print $2}'); do kill -9  $pid; done"
# echo "for pid in $(ps -ef | grep "gtamkin" | grep "Jan22" | awk '{print $2}'); do kill -9  $pid; done"
# "for pid in $(ps -ef | grep "gtamkin" | grep "Jan22" | awk '{print $2}'); do kill -9  $pid; done"