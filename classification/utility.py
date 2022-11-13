import json
import os
import re

import numpy as np
import pandas as pd

from classification.Path import ALL_URL_LIST

PATH_REGEX = {'name': r'(?P<name>\w+)',
              'dev': r'(?:(?P<dev>[^_]+)_)?',
              'sites': r'(?:(?P<sites>[^_]+)_)?',
              'date': r'(?P<date>\d\d-\d\d-\d\d)',
              'inst': r'(?:_(?P<inst>\d+))?'}
FNAME_REGEX = re.compile('{name}/{dev}{sites}{date}{inst}'.format(**PATH_REGEX))


def recover_order(sent_lengths, received_lengths, order):
    """Return sequence of lengths from snd/rcv lengths and order.

    Example:
        sent = [20, 33, 40]
        received = [33, 20, 20]
        order = [1, -1, 1, 1, -1, -1]
        Returns: [20, -33, 33, 40, -20, -20]
    """
    sequence = np.zeros(len(order))
    sequence[np.argwhere(order > 0).flatten()] = sent_lengths
    sequence[np.argwhere(order < 0).flatten()] = np.negative(received_lengths)
    return sequence


def get_bursts(len_seq):
    """Returns the sequence split by bursts.

    Example:
        len_seq = [20, -33, 33, 40, -20, -20]
        Returns: [[20], [-33], [33, 40], [-20, -20]]
    """
    directions = len_seq / abs(len_seq)
    index_dir_change = np.where(directions[1:] - directions[:-1] != 0)[0] + 1
    bursts = np.split(len_seq, index_dir_change)
    return bursts


def join_str(lengths):
    return ' '.join(map(str, lengths))


def it_webpages(fpath):
    """Iterate over all the websites contained in a file."""
    with open(fpath) as f:
        data_dict = json.loads(f.read())
        try:
            for pcap_filename, values in data_dict.items():
                webpage_num = pcap_filename[:-5]
                snd, rcv = values['sent'], values['received']
                order = values['order']
                lengths = recover_order(*map(np.array, [snd, rcv, order]))
                yield webpage_num, lengths
        except KeyError:
            print(fpath, "does not have a known order sequence")
            return
        except Exception as e:
            print("ERROR:", fpath, pcap_filename, e)


def sel_files(dpath):
    """Yield files that satisfy conditions."""
    sel_files = []
    for root, _, files in os.walk(dpath):
        for fname in files:
            if not fname.endswith('.json'):  # skip non-json files
                continue
            fpath = os.path.join(root, fname)
            sel_files.append(fpath)
    return sel_files


def load_data(dpath):
    """Traverse the directory and parse all the captures in it.

    Returns a dataframe containing encoded lengths.
    """
    print("Starting to parse")
    selected_files = sel_files(dpath)
    print("Number of selected files", len(selected_files))

    # iterave over selected files and build dataframe
    empties = 0
    idx = pd.DataFrame(columns=PATH_REGEX.keys())
    for fpath in selected_files:
        m = FNAME_REGEX.search(fpath)
        if m is None:
            print("ERROR:", fpath, FNAME_REGEX.pattern)
            continue
        row_head = {k: m.group(k) for k in PATH_REGEX}
        for i, (webpage_id, lengths) in enumerate(it_webpages(fpath)):
            if len(lengths) == 0:
                empties += 1
                continue
            row_head['fname'] = os.path.basename(fpath)
            row_head['class_label'] = webpage_id
            row_head['lengths'] = lengths
            idx = idx.append(row_head, ignore_index=True)
        print(i, 'sites in', fpath)
    print("Empty traces:", empties)

    # fix some naming issues:
    idx['inst'] = idx.inst.fillna(0)
    idx['date'] = pd.to_datetime(idx.date.str.replace('-18', '-2018'),
                                 format='%d-%m-%Y')
    return idx


def load_mapping():
    """Return Alexa as a list."""
    return [l.strip() for l in open(ALL_URL_LIST)]


ALEXA_MAP = load_mapping()
