import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from os.path import join, isdir

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from dataloaders.utils_tree import decode_segmap
from dataloaders import custom_transforms as tr
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.saver import Saver
import time
import multiprocessing
import torchvision
from DenseCRFLoss import DenseCRFLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global grad_seg 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--n_class', type=int, default=7)
    parser.add_argument('--dataset', type=str, default='tree')
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='put the path to checkpoint if needed')
    # rloss options
    parser.add_argument('--rloss-weight', type=float,
                        metavar='M', help='densecrf loss')
    parser.add_argument('--rloss-scale',type=float,default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')
    
    # output directory
    parser.add_argument('--output_directory', type=str,
                        help='output directory')

    # input image
    parser.add_argument('--image_path',type=str,default='./misc/test.png',
                        help='input image path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    print(args)
    
    # Define network
    model = DeepLab(num_classes=args.n_class,
                    backbone=args.backbone,
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)
    
    # Using cuda
    if not args.no_cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
    
    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()
    
    composed_transforms = transforms.Compose([
            # tr.FixScaleCropImage(crop_size=args.crop_size),
            tr.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensorImage()])

    green_idx=0
    total = 0
    for path, dirs, files in os.walk(args.image_path):
        for file in files:
            green = 0
            total += 1

            image_path = os.path.join(path, file)
            image_ = Image.open(image_path)
            image_size = image_.size
            image_org = image_
            image = composed_transforms(image_.convert('RGB')).unsqueeze(0)
            image_cpu = image
            if not args.no_cuda:
                image = image.cuda()
            start = time.time()


            output = model(image)
            print('inference time:',time.time()-start)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # visualize prediction
            segmap = decode_segmap(pred[0],args.dataset)*255
            segmap = segmap.astype(np.uint8)
            segimg = Image.fromarray(segmap, 'RGB')
            img_width, img_height = segimg.size

            if args.output_directory is not None:
                if not isdir(args.output_directory):
                    os.makedirs(args.output_directory)
                segimg = segimg.resize(image_size)
                segimg.save(os.path.join(args.output_directory, os.path.split(image_path)[-1]))
            else:
                plt.figure()
                plt.imshow(segimg)
                plt.show()

            # for pixel in segimg.getdata():
            #     if pixel == (128, 192, 0) or pixel == (255,255,0):
            #         green += 1
            # green_idx += (green/(img_height*img_width))

            # Add batch sample into evaluator
            # softmax = nn.Softmax(dim=1)
            # probs = softmax(output)
            # probs = Variable(probs, requires_grad=True)
            #
            # if args.rloss_weight is not None:
            #     croppings = torch.ones(pred.shape).float()
            #     if not args.no_cuda:
            #         croppings = croppings.cuda()
            #     densecrflosslayer = DenseCRFLoss(weight=args.rloss_weight, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
            #     if not args.no_cuda:
            #         densecrflosslayer.cuda()
            #     print(densecrflosslayer)
            #
            #     # resize output & image & croppings for densecrf
            #     densecrfloss = densecrflosslayer(image_cpu, probs, croppings)
            #
            #     print("densecrf loss {}".format(densecrfloss.item()))
            #
            #     # visualize densecrfloss
            #     densecrfloss.backward()
            #     #"""
            #     grad_seg = probs.grad.cpu().numpy()
            #     #print (grad_seg.shape)
            #     #print (probs.grad.sum())
            #     #print (reduced_probs.grad.sum())
            #     #grad_seg = reduced_probs.grad.cpu().numpy()
            #         # gradient of dense crf loss
            #     for i in range(args.n_class):
            #         fig=plt.figure()
            #         plt.imshow(grad_seg[0,i,:,:], cmap="hot") #vmin=0, vmax=1)
            #         plt.colorbar()
            #         plt.axis('off')
            #         if args.output_directory is not None:
            #             plt.savefig(
            #                 join(args.output_directory,args.image_path.split('/')[-1].split('.')[0]+'_grad_seg_class_' + str(i) +'.png')
            #             )
            #             plt.show(block=False)
            #             plt.close(fig)
            #     if args.output_directory is None:
            #         plt.show(block=True)

    # print(green_idx, total)
    # print("green index is {}".format(green_idx/total))

if __name__ == "__main__":
   main()
