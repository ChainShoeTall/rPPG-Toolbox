# modified from pyVHR
# https://github.com/phuselab/pyVHR
# author: Nick Chin
# copyright: (C) 2021 PanopticAI Ltd.
import cv2
import datetime
import numpy as np
from scipy import signal

import scipy.stats.kde as kde


def calc_min_interval(x, alpha):
    # FROM HDi.py file #

    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    # FROM HDi.py file #

    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


def hdi2(sample, alpha=0.05, roundto=2):
    # FROM HDi.py file #

    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    # y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []

    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (u - l) / 20  # differences of 5%
    hpd = list()
    hpd.append(round(min(hdv), roundto))

    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hpd.append(round(hdv[i - 1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
        x_hpd = x[(x > value[0]) & (x < value[1])]
        y_hpd = y[(x > value[0]) & (x < value[1])]

        modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes


class SkinDetector:
    __version__ = "1.1.3"
    strength = None
    lower1 = None
    upper1 = None
    lower2 = None
    upper2 = None
    lower = np.asarray([3, 14, 67], dtype=np.uint8)
    upper = np.asarray([16, 113, 210], dtype=np.uint8)
    stats_computed = None
    multiple_modes = None

    def __init__(self, strength=.2):
        self.strength = strength

    def compute_thresholds(self, face, fast=True):

        assert (0 <= self.strength <= 1), "'strength' parameter must have values in [0,1]"

        if fast:
            face_in = cv2.resize(face, (40, 40))
        else:
            face_in = face
        face_hsv = cv2.cvtColor(face_in, cv2.COLOR_RGB2HSV)
        h = face_hsv[:, :, 0].reshape(-1, 1)
        s = face_hsv[:, :, 1].reshape(-1, 1)
        v = face_hsv[:, :, 2].reshape(-1, 1)

        alpha = self.strength  # the highest, the stronger the masking
        hpd_h, x_h, y_h, modes_h = hdi2(np.squeeze(h), alpha=alpha)
        min_s, max_s = hdi(np.squeeze(s), alpha=alpha)
        min_v, max_v = hdi(np.squeeze(v), alpha=alpha)

        if len(hpd_h) > 1:

            self.multiple_modes = True

            if len(hpd_h) > 2:
                print('WARNING!! Found more than 2 HDIs in Hue Channel empirical Distribution... Considering only 2')
                from scipy.spatial.distance import pdist, squareform
                m = np.array(modes_h).reshape(-1, 1)
                d = squareform(pdist(m))
                maxij = np.where(d == d.max())[0]
                i = maxij[0]
                j = maxij[1]
            else:
                i = 0
                j = 1

            min_h1 = hpd_h[i][0]
            max_h1 = hpd_h[i][1]
            min_h2 = hpd_h[j][0]
            max_h2 = hpd_h[j][1]

            self.lower1 = np.array([min_h1, min_s, min_v], dtype="uint8")
            self.upper1 = np.array([max_h1, max_s, max_v], dtype="uint8")
            self.lower2 = np.array([min_h2, min_s, min_v], dtype="uint8")
            self.upper2 = np.array([max_h2, max_s, max_v], dtype="uint8")

        elif len(hpd_h) == 1:

            self.multiple_modes = False

            min_h = hpd_h[0][0]
            max_h = hpd_h[0][1]

            self.lower = np.array([min_h, min_s, min_v], dtype="uint8")
            self.upper = np.array([max_h, max_s, max_v], dtype="uint8")

        self.stats_computed = True
        # plt.figure("test"), plt.clf()
        # plt.subplot(311)
        # density_, range_, _ = plt.hist(h, density=True)
        # plt.vlines(self.lower[0], 0, np.max(density_), 'r')
        # plt.vlines(self.upper[0], 0, np.max(density_), 'r')
        # plt.subplot(312)
        # density_, range_, _ = plt.hist(s, density=True)
        # plt.vlines(self.lower[1], 0, np.max(density_), 'r')
        # plt.vlines(self.upper[1], 0, np.max(density_), 'r')
        # plt.subplot(313)
        # density_, range_, _ = plt.hist(v, density=True)
        # plt.vlines(self.lower[2], 0, np.max(density_), 'r')
        # plt.vlines(self.upper[2], 0, np.max(density_), 'r')
        # plt.show()

    def get_skin(self, face, filt_kern_size=7, verbose=False, plot=False):

        if not self.stats_computed:

            self.compute_thresholds(face, fast=True)

            # if self.lower is None or self.upper is None:
            #     raise ValueError("ERROR! You must compute stats at least one time")

        face_hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

        if self.multiple_modes:
            if verbose:
                print('\nLower1: ' + str(self.lower1))
                print('Upper1: ' + str(self.upper1))
                print('\nLower2: ' + str(self.lower2))
                print('Upper2: ' + str(self.upper2) + '\n')

            skin_mask1 = cv2.inRange(face_hsv, self.lower1, self.upper1)
            skin_mask2 = cv2.inRange(face_hsv, self.lower2, self.upper2)
            skin_mask = np.logical_or(skin_mask1, skin_mask2).astype(np.uint8) * 255

        else:

            if verbose:
                print('\nLower: ' + str(self.lower))
                print('Upper: ' + str(self.upper) + '\n')

            skin_mask = cv2.inRange(face_hsv, self.lower, self.upper)

        if filt_kern_size > 0:
            skin_mask = signal.medfilt2d(skin_mask, kernel_size=filt_kern_size)
        skin_face = cv2.bitwise_and(face, face, mask=skin_mask)

        if plot:
            h = face_hsv[:, :, 0].reshape(-1, 1)
            s = face_hsv[:, :, 1].reshape(-1, 1)
            v = face_hsv[:, :, 2].reshape(-1, 1)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.hist(h, 20)
            plt.title('Hue')
            plt.subplot(2, 2, 2)
            plt.hist(s, 20)
            plt.title('Saturation')
            plt.subplot(2, 2, 3)
            plt.hist(v, 20)
            plt.title('Value')
            plt.subplot(2, 2, 4)
            plt.imshow(skin_face)
            plt.title('Masked Face')
            plt.show()

        return skin_face

    def get_parameters(self):
        parameters = {"multiple_modes": self.multiple_modes, "stats_computed": self.stats_computed,
                      "strength": self.strength, "lower": self.lower, "upper": self.upper,
                      "lower1": self.lower1, "lower2": self.lower2, "upper1": self.upper1, "upper2": self.upper2
                      }
        return parameters

    def set_parameters(self, parameters: dict):
        for key, value in parameters.items():
            setattr(self, key, value)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # from face_detector_1_1 import FaceDetector
    from skin_detector_1_0 import SkinDetect

    sd = SkinDetect(0.15)
    sd_new = SkinDetector(.2)
    data = np.load("skin_detection_sample.npy", allow_pickle=True)
    frame = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # fd = FaceDetector()
    # face_boxes = fd.get_face_boxes(frame)
    # x1, y1, x2, y2 = [int(x) for x in face_boxes[0]]
    x1, y1, x2, y2 = 228, 135, 414, 365
    dy = y2 - y1
    y2 = int(y1 + dy * .7)
    face = frame[y1:y2, x1:x2].copy()

    for _ in range(1):
        st = datetime.datetime.now()
        sd.compute_stats(face)
        elapse = (datetime.datetime.now() - st).total_seconds()
        print("{:.1f}ms {:.2f}fps | range: {} | {} -> {}".format(elapse*1000, 1/elapse, sd.upper - sd.lower, sd.lower, sd.upper))
        # tmp = list()
        # for n in np.arange(10, 200, 10):
        #     st = datetime.datetime.now()
        #     sd.compute_stats(cv2.resize(face, (n, n)))
        #     elapse = (datetime.datetime.now() - st).total_seconds()
        #     print("{:.1f}ms {:.2f}fps | range: {} | {} -> {}".format(elapse*1000, 1/elapse, sd.upper - sd.lower, sd.lower, sd.upper))
        #     tmp.append([n, elapse])
        # tmp = np.asarray(tmp)
        # plt.figure("test"), plt.clf()
        # plt.plot(tmp[:, 0], 1 / tmp[:, 1], 'r-x')
        # plt.xlabel("face length")
        # plt.ylabel("fps")
        # plt.grid()
        # plt.show()
        st = datetime.datetime.now()
        sd.compute_stats(cv2.resize(face, (40, 40)))
        elapse = (datetime.datetime.now() - st).total_seconds()
        print("{:.1f}ms {:.2f}fps | range: {} | {} -> {}".format(elapse * 1000, 1 / elapse, sd.upper - sd.lower, sd.lower, sd.upper))
        print("-" * 8)

    st = datetime.datetime.now()
    sd_new.compute_thresholds(face, fast=True)
    elapse = (datetime.datetime.now() - st).total_seconds()
    print("compute threshold = {:.1f}ms {:.2f}fps ".format(elapse * 1000, 1 / elapse))
    print(sd_new.get_parameters())

    skin = sd.get_skin(face)
    skin_new = sd_new.get_skin(face)
    # print(sd.upper1)
    plt.figure("test", figsize=(10, 4)), plt.clf()
    plt.subplot(141)
    plt.imshow(frame)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-x')
    plt.title("Original Frame")
    plt.subplot(142), plt.imshow(face), plt.title("Face")
    plt.subplot(143), plt.imshow(skin), plt.title("pyVHR")
    plt.subplot(144), plt.imshow(skin_new), plt.title("Modified")
    plt.show()

