'''
The model with both two branches.
"Deep learning of Human Visual Sensitivity in Image Quality Assessment Framework"

'''
import torch
import torch.nn as nn
from models.util import process, per_err

torch.set_default_tensor_type('torch.FloatTensor')


class deepIQA_model(nn.Module):
    def __init__(self):
        super(deepIQA_model, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1,
                                 out_channels=32,
                                 kernel_size=(3, 3),
                                 stride=(1, 1),
                                 padding=(1, 1),
                                 bias=True)

        self.conv1_2 = nn.Conv2d(in_channels=1,
                                 out_channels=32,
                                 kernel_size=(3, 3),
                                 stride=(1, 1),
                                 padding=(1, 1),
                                 bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=32,
                                 out_channels=32,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1),
                                 bias=True)

        self.conv2_2 = nn.Conv2d(in_channels=32,
                                 out_channels=32,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1),
                                 bias=True)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)

        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(2, 2),
                               padding=(1, 1),
                               bias=True)

        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)

        self.conv6 = nn.Conv2d(in_channels=64,
                               out_channels=1,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=True)
        self.conv6.bias.data.fill_(1.)

        self.relu = nn.ReLU(inplace=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.globalpooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.ave_pooling = nn.AvgPool2d(kernel_size=(4, 4), stride=4)

        self.fc1 = nn.Linear(1, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

    def forward(self, *x):
        img, error = x[0], x[1]

        upper = self.leakyrelu(self.conv1_1(img))
        upper = self.leakyrelu(self.conv2_1(upper))

        err = error.unsqueeze(1)
        lower = self.leakyrelu(self.conv1_2(err))
        lower = self.leakyrelu(self.conv2_2(lower))

        x = torch.cat((upper, lower), dim=1)

        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.leakyrelu(self.conv5(x))
        x = self.leakyrelu(self.conv6(x)).squeeze()

        error = self.ave_pooling(error)
        # print('error', error.shape, 'x', x.shape)
        p = x*error
        print('p', p.shape)
        p = p[:, 4:-4, 4:-4]
        print('p', p.shape)
        s = self.globalpooling(p)
        s = self.leakyrelu(self.fc1(s))
        s = self.relu(self.fc2(s))
        # print('s', s.shape)
        # subjective score: s
        # sensitivity map: x

        return s.squeeze(), x


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import time
    from scipy import misc
    import numpy as np
    torch.set_num_threads(1)

    i = misc.ascent()

    imh, imw = 384, 384
    img = np.zeros((15, 1, imh, imw))
    img[0, 0, :imh, :imw] = i[:imh, :imw]
    print(img.shape)
    # test_image = Variable(torch.from_numpy(img))
    test_image = torch.tensor(torch.from_numpy(img))
    test_image = test_image.type('torch.FloatTensor')
    model = deepIQA_model()
    # On GPU
    model = model.cuda()
    test_image = test_image.cuda()


    # # Test BP
    # loss = nn.MSELoss()
    # pred = model(test_image)
    # output = loss(pred, Variable(torch.cuda.FloatTensor(1,20,imh,imw)))
    # output.backward()

    time_all = []
    for i in range(0, 100):
        t0 = time.clock()
        score, senMap = model(test_image, test_image[:, 0, :imh, :imw])

        t1 = (time.clock() - t0) * 1000
        fps = 1000 / t1

        print(i)
        print(score.shape)
        print(score.type())
        print(senMap.shape)
        # print(importance.size())
        print('Forward Time: {:.2f} ms'.format(t1))
        print('Forward FPS:  {:.2f} f/s'.format(fps))
        print('')
        time_all.append(t1)

    print('Mean Time: {:.2f} ms'.format(torch.Tensor(time_all).median()))