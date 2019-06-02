import numpy as np
import skimage.io
import skimage#, skimage.img_as_float
import scipy
import os
from glob import glob

class DataLoader():

    def __init__(self):
        self.dataset_name = "probav_data/"

    @staticmethod
    def all_scenes_paths(base_path):
        base_path = base_path if base_path[-1] in {'/', '\\'} else (base_path + '/')
        return [
            base_path + c + s
            for c in ['RED/', 'NIR/']
            for s in sorted(os.listdir(base_path + c))
            ]

    @staticmethod
    def lowres_image_iterator(path, img_as_float=True):
        path = path if path[-1] in {'/', '\\'} else (path + '/')
        for f in glob(path + 'LR*.png'):
            q = f.replace('LR', 'QM')
            l = skimage.io.imread(f, dtype=np.uint16)
            c = skimage.io.imread(q, dtype=np.bool)
            if img_as_float:
                #l = skimage.img_as_float64(l)
                l = skimage.img_as_float(l)
            yield (l, c)

    @staticmethod
    def highres_image(path, img_as_float=True):
        path = path if path[-1] in {'/', '\\'} else (path + '/')
        hr = skimage.io.imread(path + 'HR.png', dtype=np.uint16)
        sm = skimage.io.imread(path + 'SM.png', dtype=np.bool)
        if img_as_float:
            #hr = skimage.img_as_float64(hr)
            hr = skimage.img_as_float(hr)
        return (hr, sm)

    def central_tendency(self, images, agg_with='median',
                         only_clear=False, fill_obscured=False,
                         img_as_float=True):

        agg_opts = {
            'mean'   : lambda i: np.nanmean(i, axis=0),
            'median' : lambda i: np.nanmedian(i, axis=0),
            'mode'   : lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0],
            }
        agg = agg_opts[agg_with]

        imgs = []
        obsc = []

        if isinstance(images, str):
            images = self.lowres_image_iterator(images, img_as_float or only_clear)
        elif only_clear:
            images = [(l.copy(), c) for (l,c) in images]

        for (l, c) in images:

            if only_clear:
                # keep track of the values at obscured pixels
                if fill_obscured != False:
                    o = l.copy()
                    o[c] = np.nan
                    obsc.append(o)
                # replace values at obscured pixels with NaNs
                l[~c] = np.nan
            imgs.append(l)

        # aggregate the images
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            np.warnings.filterwarnings('ignore', r'Mean of empty slice')

            agg_img = agg(imgs)

            if only_clear and fill_obscured != False:
                if isinstance(fill_obscured, str):
                    agg = agg_opts[fill_obscured]
                some_clear = np.isnan(obsc).any(axis=0)
                obsc = agg(obsc)
                obsc[some_clear] = 0.0
                np.nan_to_num(agg_img, copy=False)
                agg_img += obsc

        return agg_img

    def load_data(self, batch_size=1):

        train = self.all_scenes_paths(self.dataset_name + 'train')
        batch_images = np.random.choice(train, size=batch_size)

        imgs_hr = []
        imgs_lr = []

        for img_path in batch_images:
            hr, sm = self.highres_image(img_path)
            # The SR image should not reconstruct volatile features (like clouds) or introduce artifacts.
            hr[~sm] = 0.05
            img_hr = hr #np.stack([hr,sm])

            #img_lr = self.central_tendency(img_path, agg_with='median', only_clear=True, fill_obscured=True)
            img_lr = self.central_tendency(img_path, agg_with='median', only_clear=False)

            # for future data generation
#             img_hr = np.fliplr(img_hr)
#             img_lr = np.fliplr(img_lr)

            #add channel axis
            img_hr = np.expand_dims(img_hr, axis=2)
            img_lr = np.expand_dims(img_lr, axis=2)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        return np.array(imgs_hr), np.array(imgs_lr)
