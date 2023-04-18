# -*- coding: utf-8 -*-

import os
import json
import sys


def search(dataset_path: str) -> list:
    return list(filter(lambda x: x.startswith('n'), os.listdir(dataset_path)))


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    assert os.path.exists(dataset_path), f"Path {dataset_path} does not exist."
    insts = search(sys.argv[1])
    typename = dataset_path.strip('/').split('/')[-1]


    dataset_dict = {
        'ToothMorphology': {
            typename: insts
        }
    }

    tag = typename[:typename.find(']') + 1]
    assert len(tag) > 0, f"Tag is empty for {typename}"

    with open(f'examples/splits/tooth_{tag}_train.json', 'w') as f:
        json.dump(dataset_dict, f, indent=4)

    with open(f'examples/splits/tooth_{tag}_test.json', 'w') as f:
        json.dump(dataset_dict, f, indent=4)
