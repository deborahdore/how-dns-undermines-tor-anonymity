# Extraction

In this directory is stored the script used to extract information from PCAP files (containing packet data of a
network) used to create the n-grams features for the classifier.

It works as follows:

1. Insert into the [pcap](pcap) directory the PCAP files you want to convert
2. Start the [pcap_extraction](pcaps_extraction.py) script using python
3. Wait for it to finish
4. The new dataset can be found in the [result](result) directory with today's date as name
5. Move the file to the [dataset](../../datasets) directory for the classifier to use it in the next training
    - If the PCAP files where created by making DNS request to the sites listed
      in [short_list_1500](../collection/short_list_1500), insert the new dataset into
      the [CW dataset](../../datasets/CW) that contains the dataset for the monitored sites
    - otherwise insert into [OW dataset](../../datasets/OW) that contains the dataset for the unmonitored sites
