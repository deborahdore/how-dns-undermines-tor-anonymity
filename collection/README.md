## _STEPS FOR THE DATASET COLLECTION_

1. Install VirtualBox and Vagrant
   1. Vagrant is an open-source software product for building and maintaining portable virtual software development environments
2. Set the number of VMs you want to use in line 2 of Vagrantfile 
   1. This file is basically a makefile
3. Run the command `vagrant up` 
4. Run the following command to kick off the experiment: `bash /vagrant/browse_chrome.sh N >> /vagrant/logs/$1/browse1\_$(date +%Y-%m-%d) 2>&1 &`, where N is the ID of the machine. 