import matplotlib.pyplot as plt
import json
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'A program to graph log files')
    parser.add_argument("filename", help="The name of the file to be logged (.json added at the end)")
    args = parser.parse_args()
    filename = args.filename + '.json'


    batches = json.load(open(filename, 'r'))

    for batch in batches:
        pass

    x_axis = [batch['batch'] for batch in batches]
    y_axis = [batch['reward'] for batch in batches]

    plt.plot(x_axis, y_axis, color = 'green', marker = 'x')

    plt.show()