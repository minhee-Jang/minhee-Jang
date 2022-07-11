import os
import time
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.tester import test_net_by_tensor_patches, calc_metrics, save_tensors, save_metrics, save_summary, save_results


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.is_train = False
    opt.test_random_patch = False
    opt.load_epoch = 'best'
    opt.test_ratio = 1.0
    opt.in_mem = False
    opt.multi_gpu = False
    # hard-code some parameters for test
    opt.n_threads = 0   # test code only supports num_threads = 1

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.eval()
    start_time = time.time()

    pd = opt.n_inputs

    for di, test_dataloader in enumerate(dataloader['test']):
        print("*** Test on {}***".format(opt.test_datasets[di]))
        if len(test_dataloader) == 0: continue
        avg_loss = 0.0
        avg_psnr = 0.0
        noise_avg_loss = 0.0
        noise_avg_psnr = 0.0
        mean_avg_loss = 0.0
        mean_avg_psnr = 0.0
        itr = 0
        for i, batch in enumerate(test_dataloader, 1):
            x, target, case = batch[0], batch[1], batch[2]
            # print('x.shape:', x.shape)
            # print('target.shape:', target.shape)
            # print('case:', case)
            
            _, D, h, w = x.shape
            for d in range(D - pd):
                xd = x[:, d:d+pd]
                targetd = target[:, d:d+pd]
                # print('xd.shape:', xd.shape)
                # print('targetd.shape:', targetd.shape)
                tensors_dict = {
                    "x": xd,
                    "target": targetd,
                    "case": case[0],
                }

                with torch.no_grad():
                    model.set_input(tensors_dict)
                    model.test()
                    out = model.out
                # print('out.shape:', out.shape)
                
                xd = xd.to(opt.device).detach()
                out = out.to(opt.device).detach()
                targetd = targetd.to(opt.device).detach()

                idx = '-{:03d}'.format(d+pd//2)
                fn = case[0] + idx

                tensors_dict["out"] = out
                tensors_dict["filename"] = fn
                # print('filename:', filename)

                # Show and save the results when it is testing phase

                # noise_loss, noise_psnr, mean_loss, mean_psnr, batch_loss, batch_psnr = calc_metrics(tensors_dict)
                noise_loss, noise_psnr, mean_loss, mean_psnr, batch_loss, batch_psnr = save_results(opt, di, i, fn, tensors_dict)

                noise_avg_loss += noise_loss
                noise_avg_psnr += noise_psnr
                mean_avg_loss += mean_loss
                mean_avg_psnr += mean_psnr
                avg_loss += batch_loss
                avg_psnr += batch_psnr
                itr += 1

                end_time = time.time()
                print("** Test {:.3f}s => Image({}/{})".format(end_time - start_time, i, len(test_dataloader)))
                print("Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Mean Loss: {:.8f}, Mean PSNR: {:.8f}, Batch Loss: {:.8f}, Batch PSNR: {:.8f}".format(
                     noise_loss.item(), noise_psnr.item(), mean_loss.item(), mean_psnr.item(), batch_loss.item(), batch_psnr.item()
                ))
                print("** Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Mean Loss: {:.8f}, Average Mean PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
                    noise_avg_loss / itr, noise_avg_psnr / itr, mean_avg_loss / itr, mean_avg_psnr / itr, avg_loss / itr, avg_psnr / itr
                ))
                # save_tensors(opt, di, tensors_dict)
                # save_metrics(opt, di, i, fn, noise_loss, noise_psnr, batch_loss, batch_psnr)

        avg_loss, avg_psnr = avg_loss / itr, avg_psnr / itr
        noise_avg_loss, noise_avg_psnr = noise_avg_loss / itr, noise_avg_psnr / itr
        mean_avg_loss, mean_avg_psnr = mean_avg_loss / itr, mean_avg_psnr / itr

        print("===> Test on {} - Noise Average Loss: {:.8f}, Noise Average PSNR: {:.8f}, Mean Average Loss: {:.8f}, Mean Average PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
            opt.test_datasets[di], noise_avg_loss, noise_avg_psnr, mean_avg_loss, mean_avg_psnr, avg_loss, avg_psnr
        ))
        save_summary(opt, di, noise_avg_loss, noise_avg_psnr, mean_avg_loss, mean_avg_psnr, avg_loss, avg_psnr)