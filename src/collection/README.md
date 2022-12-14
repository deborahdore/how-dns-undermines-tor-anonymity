# Data Collection

We use Vagrant with VirtualBox as provider to set up the data collection.
This guide won't work in Apple M1 processors since the version of VirtualBox for M1 is
still under development and the selected Linux Distro for this experiment only runs on Intel Processors.

In order to collect data, perform the following steps:

1. Install VirtualBox.
2. Install Vagrant.
3. Edit **Vagrantfile** to set your required number of VMs. The current configuration creates 1 VM.
4. Create two directories **pcaps** and **logs** inside the [collection](../collection) folder.
5. Run the command `vagrant up` (should be run from inside the collection folder).
6. If it doesn't run, try `vagrant reload --provision`.
7. Log into the VM with the command `vagrant ssh nodeN`, where N is the ID of the machine (N = 1, in the current
   configuration).
8. To start the experiment run `bash /vagrant/browse_chrome.sh N >> /vagrant/logs/$1/browse1\_$(date +%Y-%m-%d) 2>&1 &`,
   where N is the ID of the machine. This will create numerous pcap files containing information about packets that will
   be extracted to form the dataset.
9. Move the PCAP files into the [extraction folder](../extraction/pcap) to extract information

**Note**:

- the user must have Chrome installed