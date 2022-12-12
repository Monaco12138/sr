import argparse
import os
import math
from functools import partial
import time
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import datasets
import models
import utils


def eval_psnr(loader, model, data_norm=None, eval_dataset=None, verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_dataset is None:
        metric_fn = utils.calc_psnr

    elif eval_dataset.startswith('div2k'):
        scale = int(eval_dataset.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)

    elif eval_dataset.startswith('benchmark'):
        scale = int(eval_dataset.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)

    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    
    # inp_sub = inp_sub.half()
    # inp_div = inp_div.half()
    # gt_sub = gt_sub.half()
    # gt_div = gt_div.half()
    # model = model.half()

    index = 1
    #start = time.perf_counter()
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        # batch['inp'] = batch['inp'].half()
        # batch['gt'] = batch['gt'].half()

        inp = (batch['inp'] - inp_sub) / inp_div
        
        with torch.no_grad():
            inp_kwargs = { 'x': inp }
            pred = model( **inp_kwargs )
        
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        res = metric_fn(pred, batch['gt'])

        index = index + 1

        # nameStr = '000{}'.format(index)
        # saveImgtensor(pred[0], nameStr[-4:])

        val_res.add(res.item(), inp.shape[0])

        if verbose:                                     # 
            pbar.set_description('val {:.4f}'.format(val_res.item()))
    #end = time.perf_counter()
    #print( "infer time consumed: {}s".format(end - start) )

    return val_res.item()

def saveImgtensor( img, nameStr, savePath = '/home/ubuntu/data/main/invertible/DIV2K_SR1-900/'):
    #存储tensor格式的图像
    imgPIL = transforms.ToPILImage()( img.clip(0,1) ).convert('RGB')
    #imgPIL.save( savePath + nameStr + '.png')


if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="configs/test/test-2.yaml")
    parser.add_argument('--model', default= '/home/ubuntu/data/main/sr/save/_train_edsr_/epoch-best.pth')              # 模型存储位置
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    start = time.perf_counter()
    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_dataset=config.get('eval_dataset'),
        verbose=True)
    end = time.perf_counter()

    print( "total time consumed: {}s".format(end - start) )
    print('psnr result: {:.4f}'.format(res))
