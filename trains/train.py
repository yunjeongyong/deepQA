import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from trains.validate import val
from models.util import totalVari_regu
from datasets.TID2013Dataset import TID2013Dataset


def snapshot(model, testloader, epoch, best, snapshot_dir, prefix, is_first):
    val_Dict = val(model, testloader, is_first)
    lcc = val_Dict['lcc']
    srocc = val_Dict['srocc']
    test_loss = val_Dict['test_loss']

    snapshot = {
        'epoch': epoch,
        'model': model.module.state_dict(),
        'lcc': lcc,
        'srocc': srocc
    }

    if lcc + srocc >= best:
        best = lcc + srocc
        torch.save(snapshot, os.path.join(snapshot_dir, '%s_%.4f_%.4f_epoch%d.pth' %
                                          (prefix, lcc, srocc, epoch)))

    torch.save(snapshot, os.path.join(snapshot_dir, '{0}.pth'.format(prefix)))

    print("[{}] Curr LCC: {:0.4f} SROCC: {:0.4f}".format(epoch, lcc, srocc))

    out_dict = {'lcc': lcc,
                'srocc': srocc,
                'best': best,
                'test_loss': test_loss,
                'pred': val_Dict['pre_array'],
                'gt': val_Dict['gt_array'],
                'img': val_Dict['img'],
                'error': val_Dict['error'],
                'senMap': val_Dict['senMap']
                }

    return out_dict




def trainProcess(model, optimG, trainloader, testloader, max_epoch, snapshot_dir, prefix, is_first):
    best_lcc = -1
    # weight = torch.cuda.FloatTensor([0.5, 1.0])

    for epoch in range(1, max_epoch + 1):
        loss_score = []
        model.train()

        for batch_id, (img, ref, error, dmos) in enumerate(trainloader):
            dmos = dmos.type('torch.FloatTensor')
            # img, error, dmos, = Variable(img.cuda()), \
            #                         Variable(error.cuda()), \
            #                         Variable(dmos.cuda())

            img = torch.tensor(img, device='cuda', requires_grad=True, dtype=torch.float)
            error = torch.tensor(error, device='cuda', requires_grad=True, dtype=torch.float)
            dmos = torch.tensor(dmos, device='cuda', requires_grad=True, dtype=torch.float)

            optimG.zero_grad()

            score_pred, senMap = model(img, error)
            # print('train', score_pred.shape, score_gt.shape)
            criterion = nn.MSELoss()
            loss_1 = criterion(score_pred, dmos)
            tv_reg = torch.mean(totalVari_regu(senMap))

            # Loss function
            LGseg = 1000*loss_1+0.01*tv_reg
            tmpscore = LGseg.data

            LGseg.backward()
            itr = len(trainloader) * (epoch - 1) + batch_id
            # poly_lr_scheduler(optimG, args.lr, itr)
            optimG.step()

            loss_score.append(tmpscore)
            # loss_importance.append(tmpimp)
            print("[{0}][{1}] ScoreL1: {2:.4f} TVreg: {3:.2f}."
                  .format(epoch, itr, tmpscore, tv_reg))

        if epoch % 10 == 0:
            snap_dict = snapshot(model,
                                 testloader,
                                 epoch,
                                 best_lcc,
                                 snapshot_dir,
                                 prefix,
                                 is_first)
            best_lcc = snap_dict['best']
