import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
# from torchvision.utils import make_grid
from models.deepQA import deepIQA_model as predictNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from datasets.custom_dataset import Kadid10kDataset
from datasets.TID2013Dataset import TID2013Dataset
from trains.train import trainProcess


if __name__ == '__main__':
    is_first = True
    lr = 0.01
    batch_size = 16
    max_epoch = 3000
    step = 1
    snapshot_dir = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\snapshot'
    data_type = 'tid'

    # transforms_train = transforms.Compose([transforms.Resize((112, 112)),
    #                                        transforms.ToTensor()])
    #
    # transforms_test = transforms.Compose([transforms.Resize((112, 112)),
    #                                       transforms.ToTensor()])

    csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\all_data_csv\\TID2013.txt.csv'
    data_path = 'C:\\Users\\yunjeongyong\\Desktop\\DeepQA-yunjeong\\data\\TID2013_dataset'
    trainset = TID2013Dataset(csv_path, data_path, is_train=True, return_type='all', test_data=data_type)
    testset = TID2013Dataset(csv_path, data_path, is_train=False, return_type='all',
                              test_data=data_type)
    # trainset = Kadid10kDataset(transforms=transforms_train, is_train=True, return_type='all', test_data=data_type)
    # testset = Kadid10kDataset(transforms=transforms_test, is_train=False, return_type='all', test_data=data_type)

    train_batch_size = batch_size

    trainloader = DataLoader(trainset,
                             batch_size=train_batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            shuffle=True,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    model = predictNet()

    # optimG = optim.SGD(filter(lambda p: p.requires_grad, \
    #     model.parameters()),lr=args.lr,momentum=0.9,\
    #     weight_decay=1e-4,nesterov=True)
    optimG = optim.Adam(filter(lambda p: p.requires_grad,
                               model.parameters()), lr=lr, weight_decay=5e-3)

    # model = model.cuda()
    model = nn.DataParallel(model).cuda()
    trainProcess(
        model,
        optimG,
        trainloader,
        testloader,
        max_epoch,
        snapshot_dir,
        data_type,
        is_first
    )
