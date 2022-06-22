from tqdm import tqdm
from PIL import Image
import csv


class Kadid10kData:
    def __init__(self, limit=None):
        csv_path = './kadid10k/dmos.csv'
        folder = './kadid10k/images/'
        ref_folder = './kadid10k/ref/'

        self.limit = limit
        self.ref = []
        self.dist = []
        self.X = []
        self.dmos = []

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            if self.limit is None:
                self.limit = len(rows)
            for dist, ref, dmos, _ in tqdm(rows, total=self.limit):
                dist_img = Image.open(folder + dist)
                dist_img = dist_img.convert("RGB")

                ref_img = Image.open(ref_folder + ref)
                ref_img = ref_img.convert("RGB")

                # [(dist, ref), (dist, ref) ... ]
                self.X.append((dist_img, ref_img))
                self.ref.append(ref_img)
                self.dist.append(dist_img)
                self.dmos.append(float(dmos))

                if idx == self.limit:
                    break
                else:
                    idx += 1


class LiveData:
    def __init__(self, limit=None):
        csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\all_data_csv\\LIVE.txt.csv'
        folder = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\LIVE_dataset\\databaserelease2'

        self.limit = limit
        self.X = []
        self.dist = []
        self.ref = []
        self.dmos = []

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            # if self.limit is None:
            #     self.limit = len(rows)
            self.limit = len(rows)
            for dist, _, ref, dmos in tqdm(rows, total=self.limit):
                dist_img = Image.open(folder + dist)
                dist_img = dist_img.convert("RGB")

                ref_img = Image.open(folder + ref)
                ref_img = ref_img.convert("RGB")

                self.X.append((dist_img, ref_img))
                self.dist.append(dist_img)
                self.ref.append(ref_img)
                self.dmos.append(float(dmos))

                if idx == self.limit:
                    break
                else:
                    idx += 1


class TidData:
    def __init__(self, limit=None):
        csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\all_data_csv\\TID2013.txt.csv'
        folder = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\TID2013_dataset'

        self.limit = limit
        self.X = []
        self.dist = []
        self.ref = []
        self.dmos = []
        self.dmos_norm = []

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            if self.limit is None or self.limit > len(rows):
                self.limit = len(rows)
            for dist, _, ref, dmos in tqdm(rows, total=self.limit):
                dist_img = Image.open(folder + dist)
                dist_img = dist_img.convert("RGB")

                ref_img = Image.open(folder + ref)
                ref_img = ref_img.convert("RGB")

                self.X.append((dist_img, ref_img))
                self.dist.append(dist_img)
                self.ref.append(ref_img)
                self.dmos.append(float(dmos))

                if idx == self.limit:
                    break
                else:
                    idx += 1
