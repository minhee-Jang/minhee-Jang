import os, sys
from skimage.io import imsave
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions
from deltaMovingmap import EstimationDelta

if __name__ == '__main__':
    opt = TestOptions(r'D:\\data\\video-sr').parse()
    n_frames = opt.n_frames     # 5
    dataloader = create_dataset(opt)
    move_thr = opt.move_thr
    delta = EstimationDelta()

    # Make directories for output imgs
    test_dir = opt.test_results_dir + '0623threshold' + str(move_thr) 
    os.makedirs(test_dir, exist_ok=True)
    mfcnn_dir = os.path.join(test_dir, 'mfcnn')
    os.makedirs(mfcnn_dir, exist_ok=True)
    recursive_dir = os.path.join(test_dir, 'recursive')
    os.makedirs(recursive_dir, exist_ok=True)
    combine_dir = os.path.join(test_dir, 'combine')
    os.makedirs(combine_dir, exist_ok=True)
    deltaout_dir = os.path.join(test_dir, 'deltaout')
    os.makedirs(deltaout_dir, exist_ok=True)
    threout_dir = os.path.join(test_dir, 'threout')
    os.makedirs(threout_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xAll = batch['lr']    #low
        rfAll = batch['hr']    #reculsive output
        mfAll = batch['mf']    #mfcnn output
        bs, c, n, h, w = xAll.shape
        
        # 000~004는 DL 결과가 없으므로 RF 결과를 내보냄
        for j in range(n_frames):
            print('j:', j)
            fn_center = '{:03d}'.format(j)
            
            # RF 결과 가져옴
            rf = rfAll[:, :, j]
            
            # Save output imgs
            cout = rf.to('cuda')      # Combine output
            cout = cout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            combine_path = os.path.join(combine_dir, fn_center + '.tiff')
            imsave(combine_path, cout)

        # 005~191는 delta로 무빙맵을 구해 DL과 RF combine
        for j in range(n_frames, n):
            print('j:', j)
            fn_center = '{:03d}'.format(j)
            
            x = xAll[:, :, j-n_frames:j+1]   # 000~191 (delta구할때000부터필요)
            mf = mfAll[:, :, j]     # 004~191
            rf = rfAll[:, :, j]     # 004~191
            
            # Calculate delta moving map and combine DL & RF
            mout, rout, cout, dout, tout = delta(x, rf, mf, move_thr, n_frames)
            
            # Save output imgs
            mout = mout.to('cuda')      # MFCNN output
            mout = mout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            mfcnn_path = os.path.join(mfcnn_dir, fn_center + '.tiff')
            imsave(mfcnn_path, mout)
            rout = rout.to('cuda')      # Recursive Filter output
            rout = rout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(recursive_dir, fn_center + '.tiff')
            imsave(recursive_path, rout)
            cout = cout.to('cuda')      # Combine output
            cout = cout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            combine_path = os.path.join(combine_dir, fn_center + '.tiff')
            imsave(combine_path, cout)
            dout = dout.to('cuda')      # Moving map(Delta)
            dout = dout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            delta_path = os.path.join(deltaout_dir, fn_center + '.tiff')
            imsave(delta_path, dout)
            tout = tout.to('cuda')      # Moving map(Threshold)
            tout = tout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            thre_path = os.path.join(threout_dir, fn_center + '.tiff')
            imsave(thre_path, tout)