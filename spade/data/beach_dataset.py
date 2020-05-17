"""
Proof of concept - Beach dataset extracted from ADE20K 2016 Dataset
Author: Chunze Fu
"""



import os.path
from spade.data.pix2pix_dataset import Pix2pixDataset
from spade.data.image_folder import make_dataset

class BeachDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):

        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        # parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=72)
        else:
            parser.set_defaults(load_size=7)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=20)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        parser.set_defaults(aspect_ratio=1)
        parser.set_defaults(batchSize=8)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(contain_dontcare_label=False)
        return parser


    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root, '%s_label' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = os.path.join(root, '%s_img' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        # there is no instance map for beach dataset
        if not opt.no_instance:
            instance_paths = []
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
