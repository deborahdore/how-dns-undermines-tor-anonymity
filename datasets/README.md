# Dataset directory

This directory contains the datasets collected and used for the experiment.

- the [Closed World CW](CW) folder contains information extracted from the DNS requests of monitored websites - the ones
  contained in
  the [short_list_1500](../src/collection/short_list_1500). Monitored websites are identified with a number from 1 to
  1500 that correspond to the index of the websited in the [short_list_1500](../src/collection/short_list_1500).
- the [Open World OW](OW) folder contains information extracted from the DNS requests of unmonitored websites. These
  websites are identified with a number that's over 1500.

*TO ADD WEBSITES*: <br>
After collection all the pcaps file with the instruction in the [collection](/src/collection) folder, the
script *[pcaps_extraction](../src/extraction/pcaps_extraction.py)* can be used to create the datasets. 
