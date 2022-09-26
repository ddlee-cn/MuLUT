import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
from common.utils import PSNR, cal_ssim, modcrop, _rgb2ycbcr


# 4D equivalent of triangular interpolation, faster version
def FourSimplexInterpFaster(weight, img_in, h, w, interval, rot, upscale=4, mode='s'):
    q = 2 ** interval
    L = 2 ** (8 - interval) + 1

    if mode == "s":
        # Extract MSBs
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, 1:1 + h, 1:1 + w] % q

    elif mode == 'd':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, 2:2 + h, 2:2 + w] % q

    elif mode == 'y':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, 2:2 + h, 1:1 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)

    p0000 = p0000.reshape(sz, -1)
    p0100 = p0100.reshape(sz, -1)
    p1000 = p1000.reshape(sz, -1)
    p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)

    p0001 = p0001.reshape(sz, -1)
    p0101 = p0101.reshape(sz, -1)
    p1001 = p1001.reshape(sz, -1)
    p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)

    p0010 = p0010.reshape(sz, -1)
    p0110 = p0110.reshape(sz, -1)
    p1010 = p1010.reshape(sz, -1)
    p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)

    p0011 = p0011.reshape(sz, -1)
    p0111 = p0111.reshape(sz, -1)
    p1011 = p1011.reshape(sz, -1)
    p1111 = p1111.reshape(sz, -1)

    fab = fa > fb;
    fac = fa > fc;
    fad = fa > fd

    fbc = fb > fc;
    fbd = fb > fd;
    fcd = fc > fd

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
        (img_a1.shape[0], img_a1.shape[1] * upscale, img_a1.shape[2] * upscale))
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    return out


class eltr:
    def __init__(self, dataset, opt, lutDict):
        folder = os.path.join(opt.testDir, dataset, 'HR')
        files = os.listdir(folder)
        files.sort()

        exp_name = opt.expDir.split("/")[-1]
        result_path = os.path.join(opt.resultRoot, exp_name, dataset, "X{}".format(opt.scale))
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.opt = opt
        self.lutDict = lutDict

    def run(self, num_worker=24):
        pool = Pool(num_worker)
        psnr_ssim_s = pool.map(self._worker, list(range(len(self.files))))
        print('Dataset {} | AVG LUT PSNR: {:.2f} SSIM: {:.4f}'.format(dataset, np.mean(np.asarray(psnr_ssim_s)[:, 0]),
                                                                      np.mean(np.asarray(psnr_ssim_s)[:, 1])))

    def _worker(self, i):
        # Load LR image
        img_lr = np.array(Image.open(
            os.path.join(self.opt.testDir, dataset, 'LR_bicubic/X'.format(self.opt.scale), self.files[i]))).astype(
            np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)
        # Load GT image
        img_gt = np.array(Image.open(os.path.join(self.opt.testDir, dataset, 'HR', self.files[i])))
        img_gt = modcrop(img_gt, self.opt.scale)

        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)

        for s in range(self.opt.stages):
            pred = 0
            if (s + 1) == self.opt.stages:
                upscale = opt.scale
                avg_factor, bias = len(self.opt.modes), 0
            else:
                upscale = 1
                avg_factor, bias = len(self.opt.modes) * 4, 127
            for mode in self.opt.modes:
                key = "s{}_{}".format(str(s + 1), mode)
                if mode in ["d", "y"]:
                    pad = (0, 2)
                else:
                    pad = (0, 1)
                for r in [0, 1, 2, 3]:
                    img_lr_rot = np.rot90(img_lr, r)
                    h, w, _ = img_lr_rot.shape
                    img_in = np.pad(img_lr_rot, (pad, pad, (0, 0)), mode='edge').transpose((2, 0, 1))
                    pred += FourSimplexInterpFaster(self.lutDict[key], img_in, h, w, self.opt.interval, 4 - r,
                                                    upscale=upscale, mode=mode)

            img_lr = np.clip((pred / avg_factor) + bias, 0, 255)
            img_lr = img_lr.transpose((1, 2, 0))
            img_lr = np.round(np.clip(img_lr, 0, 255))
            if (s + 1) == self.opt.stages:
                img_lr = img_lr.astype(np.uint8)
            else:
                img_lr = img_lr.astype(np.float32)

        # Save to file
        img_out = img_lr
        Image.fromarray(img_out).save(
            os.path.join(self.result_path, '{}_{}_{}bit.png'.format(self.files[i].split('/')[-1][:-4], self.opt.lutName,
                                                                    8 - self.opt.interval)))
        y_gt, y_out = _rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0]
        psnr = PSNR(y_gt, y_out, self.opt.scale)
        ssim = cal_ssim(y_gt, y_out)
        return [psnr, ssim]


if __name__ == "__main__":
    opt = TestOptions().parse()

    # Load LUT
    lutDict = dict()
    for s in range(opt.stages):
        if (s + 1) == opt.stages:
            v_num = opt.scale * opt.scale
        else:
            v_num = 1
        for mode in opt.modes:
            key = "s{}_{}".format(str(s + 1), mode)
            lutPath = os.path.join(opt.expDir,
                                   "{}_x{}_{}bit_int8_{}.npy".format(opt.lutName, opt.scale, 8 - opt.interval, key))
            lutDict[key] = np.load(lutPath).astype(np.float32).reshape(-1, v_num)

    # all_datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
    all_datasets = ['Set5']

    for dataset in all_datasets:
        etr = eltr(dataset, opt, lutDict)
        etr.run()

# Reference results:
# Dataset Set5 | AVG LUT PSNR: 30.61 SSIM: 0.8655
# Dataset Set14 | AVG LUT PSNR: 27.60 SSIM: 0.7544
# Dataset B100 | AVG LUT PSNR: 26.86 SSIM: 0.7112
# Dataset Urban100 | AVG LUT PSNR: 24.46 SSIM: 0.7196
# Dataset Manga109 | AVG LUT PSNR: 27.92 SSIM: 0.8637
