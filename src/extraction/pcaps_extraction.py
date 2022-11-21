import json

from scapy.all import *
from scapy.layers.l2 import Ether

# MAC address from where dns request where performed
MY_MAC = "08:00:27:97:3f:45"
# directory where pcap files are stored
PCAP_DIR = "./pcap"
# file where resulting dataset is saved
RESULT_FILE = "../result_PCAP_extraction.json"


def direction(pkt_dir):
    """
    If the source MAC address of the packet is not my MAC address, then the packet is incoming, otherwise it's outgoing

    :param pkt_dir: the packet
    :return: The direction of the packet.
    """

    if pkt_dir[Ether].src != MY_MAC:
        # incoming
        return -1
    else:
        # outgoing
        return 1


def get_size(pkt):
    """
    It returns the length of the packet

    :param pkt: the packet to be analyzed
    :return: The length of the packet.
    """
    return len(pkt)


def create_df(source, target):
    """
    It takes a source directory and a target file, processes file in the source directory and creates a new file - taget -
    with the resulting dataset

    :param source: The directory where the pcap files are located
    :param target: The name of the resulting file you want to create
    """
    y = {}
    os.chdir(source)
    # take all pcap files in directory
    for file in glob('*.pcap'):

        received = []
        sent = []
        order = []

        dnsPackets = rdpcap(file)

        for pkt in dnsPackets:

            if not pkt.haslayer(Ether):
                break

            direction_pkt = direction(pkt)
            order.append(direction_pkt)

            if direction_pkt == -1:
                received.append(get_size(pkt))
            else:
                sent.append(get_size(pkt))

        x = {
            f"{os.path.basename(file)}": {
                "received": received,
                "order": order,
                "sent": sent
            }
        }
        y.update(x)

    with open(target, mode='w', encoding='utf-8') as result_file:
        json.dump(y, result_file)


if __name__ == '__main__':
    create_df(PCAP_DIR, RESULT_FILE)
