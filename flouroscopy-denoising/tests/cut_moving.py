import os, glob
import imageio

def cut_tiff(src_dir, ps):
    dst_dir = src_dir + '{}'.format(ps)
    src_list = glob.glob(os.path.join(src_dir, '**', '*.tiff'), recursive=True)

    for sp in src_list:
        dose, case, fn = sp.split('\\')[-3:]
        print('dose: {}, case: {}, fn: {}'.format(dose, case, fn))

        
        if 'hand' in sp:
            xs, ys = 230, 480
            xe, ye = xs + ps, ys + ps
        else:
            xs, ys = 150, 540
            xe, ye = xs + ps, ys + ps

        tiff = imageio.imread(sp)
        cut = tiff[ys:ye, xs:xe]

        dd = os.path.join(dst_dir, dose, case)
        os.makedirs(dd, exist_ok=True)
        dp = os.path.join(dd, fn)
        print(dp)
        imageio.imsave(dp, cut)
        
if __name__ == '__main__':
    train_dir = r'D:\data\flouroscopy-denoising\train\moving'
    test_dir = r'D:\data\flouroscopy-denoising\test\moving'

    # dst_train_dir = r'D:\data\flouroscopy-denoising\train\moving_cut'
    # dst_test_dir = r'D:\data\flouroscopy-denoising\test\moving_cut'

    train_list = glob.glob(os.path.join(train_dir, '**', '*.tiff'), recursive=True)
    test_list = glob.glob(os.path.join(test_dir, '**', '*.tiff'), recursive=True)

    ps = 700

    cut_tiff(train_dir, ps)
    cut_tiff(test_dir, ps)