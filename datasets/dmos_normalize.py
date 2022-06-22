import csv


def minmax_normalize(x, min, max):
    return (x - min) / (max - min)


if __name__ == '__main__':
    with open('./kadid10k/dmos.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        dmos_array = [float(dmos) for _, _, dmos, _ in reader]
        dmos_min = min(dmos_array)
        dmos_max = max(dmos_array)
        dmos_normalized_array = [minmax_normalize(dmos, dmos_min, dmos_max) for dmos in dmos_array]

        for dmos in dmos_normalized_array:
            print(dmos)
