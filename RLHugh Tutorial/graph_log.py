import matplotlib.pyplot as plt
import json
import argparse

"""
He kinda just did all of this lmao
"""


def run(args):
    log_rows = []
    with open(args.in_logfile) as f:
        for line in f:
            row = json.loads(line)
            log_rows.append(row)
    print("log rows: ", log_rows)

    # Graph

    episodes = [row['episode']for row in log_rows]
    losses = [row['loss']for row in log_rows]

    plt.plot(episodes, losses)
    plt.savefig('graph.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-logfile', type=str, default='log.txt')
    args = parser.parse_args()
    run(args)