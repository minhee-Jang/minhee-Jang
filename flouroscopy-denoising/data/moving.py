import os
import glob

from data.patchdata2 import PatchData2

class Moving(PatchData2):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(rgb_range=1.0)
        return parser
        
    def __init__(self, args, name='moving', is_train=True):
        super(Moving, self).__init__(
            args, name=name, is_train=is_train
        )

    def _scan(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**' , '*' + self.ext[1]))
        )
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )

        lr_dict = {}
        hr_dict = {}

        case_list = sorted(
            os.listdir(self.dir_lr)
        )

        for case in case_list:
            lr_dict[case] = sorted(
                glob.glob(os.path.join(self.dir_lr, case , '*' + self.ext[1]))
            )
            hr_dict[case] = sorted(
                glob.glob(os.path.join(self.dir_hr, case, '*' + self.ext[0]))
            )
            # print('case:', case)
            # for f_lr in lr_dict[case]:
            #     print(f_lr)
            # print('----')
            
        # return names_hr, names_lr
        # print('names_hr:', names_hr)
        # print('names_lr:', names_lr)
        # print('case_list:', case_list)
        # print('hr_dict:', hr_dict)
        # print('lr_dict:', lr_dict)
        return names_hr, names_lr, case_list, hr_dict, lr_dict

    def _set_filesystem(self, data_dir):
        # super(Moving, self)._set_filesystem(data_dir)
        self.apath = os.path.join(data_dir, self.mode, self.dataset)
        self.dir_hr = os.path.join(self.apath, 'high')
        self.dir_lr = os.path.join(self.apath, 'low')
        self.ext = ('.tiff', '.tiff')
        # print("dir_hr:", self.dir_hr)