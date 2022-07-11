import os, glob

if __name__ == '__main__':
    gp = r'F:\data\genoray-moving-tiff-sync'
    dst_gp = r'F:\data\genoray-moving-tiff-sync2'

    high_gl = glob.glob(os.path.join(gp, 'high', '*', '*.tiff'))
    low_gl = glob.glob(os.path.join(gp, 'low', '*', '*.tiff'))

    for hp in high_gl:
        print(hp)
        hp.split('\\')

    for lp in low_gl:
        print(lp)

