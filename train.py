import zrcrm_data
import zrcrm_loss
import torch.utils.data
from model import My_Model
import torch.optim
import os
import argparse
import pyiqa

def train(args):
    mymodel = My_Model()
    mymodel.to(args.device)
    if args.ckp:
        mymodel.load_state_dict(torch.load(args.ckp))
    else:
        for m in mymodel.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.2)
                torch.nn.init.constant_(m.bias.data, 0.0)

    train_subfolders =\
        ["train",
         "exptar",
         "adaptivesmoothtar"]
    zrcrm_data.prepare_assist_datasets(args.train_dir, device=args.device, hparams={'train_subfolders': train_subfolders})

    train_dataset = zrcrm_data.Multi_input_dataset(data_path=args.train_dir,
                                                   train_folders=train_subfolders)
    print("Total train examples:", train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.num_workers)

    if args.valid_dir:
        val_dataset = zrcrm_data.My_dataset(data_path=args.valid_dir,
                                                    with_ref=True,
                                                    in_path="Low",
                                                    tar_path="Normal",
                                                    filename_pattern="*")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True,
                                               num_workers=args.num_workers)
        ssim_metric = pyiqa.create_metric('ssim')

    loss = zrcrm_loss.My_loss()

    logvars = []
    for i in range(len(loss.all_loss_name)):
        logvars.append(torch.nn.Parameter(torch.zeros((1,), requires_grad=True).to(args.device)))
    opt_parameters = [p for p in mymodel.parameters()] + logvars if loss.hparams['auto_multi_task_loss'] else mymodel.parameters()
    optimizer = \
        torch.optim.Adam(opt_parameters, lr=args.learning_rate)

    end_factor = 1.0
    start_factor = 0.01
    warm_up_iters = 100
    lambda0 = lambda cur_iter: ((end_factor - start_factor) / warm_up_iters * cur_iter + start_factor)
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda0)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                      T_0=200,
                                                                      T_mult=1,
                                                                      eta_min=args.learning_rate/10 )
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2],
                                                      milestones=[warm_up_iters])
    # the first step of scheduler is base learning rate, skip it
    scheduler.step()

    mymodel.train()
    for epoch in range(args.num_epochs):
        print('current epoch index: ', epoch)
        for iteration, batch in enumerate(train_loader):
            total_iteration = iteration+epoch*train_loader.__len__()
            batch_data = batch[train_subfolders[0]].to(args.device)
            exptar = batch[train_subfolders[1]].to(args.device)
            adaptivesmoothtar = batch[train_subfolders[2]].to(args.device)

            enhanced, K_map = mymodel(batch_data)
            temp_loss = loss(P1=enhanced, exptar=exptar, adaptivesmoothtar=adaptivesmoothtar,
                             logvars=logvars)
            # print("total_iteration: ", total_iteration, "loss:", temp_loss.item(), "LOSSweights", [torch.exp(-1*i).item() for i in logvars])

            optimizer.zero_grad()
            temp_loss.backward()
            optimizer.step()
            scheduler.step()

            if (total_iteration+1) % args.snapshots_iter == 0:
                torch.save(mymodel.state_dict(), args.snapshots_folder + "Epoch" + str(epoch) + "batch" + str(iteration) + '.pth')
                print("saving ckp as args.snapshots_folder" + "Epoch" + str(epoch) + "batch" + str(iteration) + '.pth')
            if args.valid_dir and (total_iteration+1) % args.valid_iter == 0:
                print("validating model...")
                ssim = []
                for _, batch in enumerate(val_loader):
                    in_img, tar_img = batch
                    in_img = in_img.to(args.device)
                    tar_img = tar_img.to(args.device)
                    with torch.no_grad():
                        enhanced, K_map = mymodel(in_img)
                        ssim.append(ssim_metric(enhanced, tar_img))
                ssim = torch.mean(torch.stack(ssim, dim=0))
                print("total_iteration: ", total_iteration, "ssim:", ssim)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ckp', type=str, default=None)
    parser.add_argument('--train_dir', type=str, default='train_data')
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=int, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default='snapshots/')
    parser.add_argument('--snapshots_iter', type=int, default=10)
    parser.add_argument('--valid_dir', type=str, default='valid_data')
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--valid_iter', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.snapshots_folder):
        os.mkdir(args.snapshots_folder)

    train(args)









