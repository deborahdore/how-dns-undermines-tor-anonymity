from scapy.all import *

if __name__ == '__main__':
    data = "568.pcap"
    a = rdpcap(data)
    sessions = a.sessions()
    print(sessions)
