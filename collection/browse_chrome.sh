#/usr/bin/bash
while true; do
	d=`date "+%d-%m-%y-%H%M%S"`
	mkdir /vagrant/pcaps/$1
	mkdir /vagrant/pcaps/$1/$d
	sudo systemctl stop cloudflared
	sleep 3
	for i in {0..1499}
	do
		echo $i
		sudo /usr/sbin/tcpdump -i eth0 host 1.1.1.1 and port 443 -w /vagrant/pcaps/$1/$d/$i.pcap &
		sleep 2
		sudo systemctl start cloudflared
		sleep 2
		python /vagrant/chrome_driver.py $i
		sleep 2
		sudo systemctl stop cloudflared
		sleep 2
		sudo pkill tcpdump
		sleep 2
	done
	sleep 300
	sudo systemctl start cloudflared
done
