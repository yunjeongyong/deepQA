import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.stats


def val(model, testloader, is_first=True):
    model.eval()
    results = []
    test_loss = []

    for img_id, (img, ref, error, dmos) in enumerate(testloader):
        dmos = dmos.type('torch.FloatTensor')
        # img, error = Variable(img.cuda()), Variable(error.cuda())
        # img, error = torch.tensor(img.cuda()), torch.tensor(error.cuda())
        img = torch.tensor(img, device='cuda', requires_grad=False, dtype=torch.float)
        error = torch.tensor(error, device='cuda', requires_grad=False, dtype=torch.float)
        # score_gt = score_gt.cuda()

        score_pred, sensitivity_map = model(img, error)

        score_pred = score_pred.data.cpu()

        for k in range(dmos.size(0)):

            final_score = score_pred
            # print(score_gt[k], final_score)
            results.append([dmos[k], final_score])

            loss = 1000*nn.MSELoss()(dmos[k], final_score)
            test_loss.append(loss)

    # results = np.array(results, dtype=float)
    results = torch.as_tensor(results, dtype=float)
    lcc = np.corrcoef(results, rowvar=False)[0][1]
    srocc = scipy.stats.spearmanr(results[:, 0], results[:, 1])[0]

    img = img.squeeze()
    error = error.squeeze()
    sensitivity_map = sensitivity_map.squeeze()
    return {
        'lcc': lcc,
        'srocc': srocc,
        'test_loss': np.mean(test_loss),
        'pre_array': results[:, 0],
        'gt_array': results[:, 1],
        'img': img.data.cpu().numpy(),
        'error': error.data.cpu().numpy(),
        'senMap': sensitivity_map.data.cpu().numpy()
    }

