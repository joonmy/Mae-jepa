import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=1024)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    
    setup_seed(args.seed)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae+jepa-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_JEPA_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    save_frequency = 100

    ema = [0.996, 1.0]
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(len(dataloader)*args.total_epoch*ipe_scale)
                        for i in range(int(len(dataloader)*args.total_epoch*ipe_scale)+1))

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        losses_image = []
        losses_latent = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask, predict_latent, mask_target, target_features = model(img)
            
            # print("image loss: ",((predicted_img - img) ** 2 * mask).sum())
            # print("image loss shape: ",((predicted_img - img) ** 2 * mask).shape)
            # 512, 3, 32, 32
            loss_image = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio

            # print("latent loss: ",((predict_latent - target_features) ** 2 * mask_target).sum())
            # print("latent loss shape: ",((predict_latent - target_features) ** 2 * mask_target).shape)
            # 256, 512, 196
            loss_latent = torch.mean((predict_latent - target_features) ** 2 * mask_target) / args.mask_ratio

            loss = loss_image + loss_latent

            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            losses_image.append(loss_image.item())
            losses_latent.append(loss_latent.item())
            
            model.update_target(next(momentum_scheduler)) ## target update by EMA

            
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        avg_loss_image = sum(losses_image) / len(losses_image)
        avg_loss_latent = sum(losses_latent) / len(losses_latent)

        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        writer.add_scalar('mae_loss_image', avg_loss_image, global_step=e)
        writer.add_scalar('mae_loss_latent', avg_loss_latent, global_step=e)

        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask, _, _, _ = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        if e % save_frequency == 0 | e == args.total_epoch-1:
            torch.save(model, args.model_path+"_epoch_"+str(e))