import math
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, patches
from matplotlib.patches import ConnectionPatch
from sklearn.cluster import DBSCAN

from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import LoFTR, default_cfg


def get_matches_loftr(img0, img1, resolution=(640, 480)):
    """
    Given a pair of images, it returns the matched pairs via LOFTR. Changing
    pair order may change results.
    :param numpy.ndarray img0: array representation of img0
    :param numpy.ndarray img1: array representation of img1
    :param tuple resolution: image resolution at which images are compared
    :rtype: object
    """
    # initialize results object
    res = {}

    # initialize LoFTR
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cpu()

    # load images, resize, convert to float
    img0_resized = cv2.resize(img0, resolution)
    img1_resized = cv2.resize(img1, resolution)
    img0_f = torch.from_numpy(img0_resized)[None][None].cpu() / 255.
    img1_f = torch.from_numpy(img1_resized)[None][None].cpu() / 255.
    batch = {'image0': img0_f, 'image1': img1_f}

    # inference
    with torch.no_grad():
        matcher(batch)    # batch = {'image0': img0, 'image1': img1}
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # rescale mkpts to original scale
    mkpts0_org = mkpts0 * [img0.shape[1] / img0_resized.shape[1], img0.shape[0] / img0_resized.shape[0]]
    mkpts1_org = mkpts1 * [img1.shape[1] / img1_resized.shape[1], img1.shape[0] / img1_resized.shape[0]]

    # Draw
    plt.ioff()
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(img0, img1, mkpts0_org, mkpts1_org, color, mkpts0_org, mkpts1_org, text)
    plt.close(fig)
    plt.ion()

    # build results object
    res['img0'] = img0          # original image
    res['img1'] = img1
    res['figure'] = fig         # figure of matches
    res['mkpts0'] = mkpts0_org  # mkpts scaled to original image
    res['mkpts1'] = mkpts1_org

    return res


def find_xy_boundary(pts):
    # finds a bounding box for cluster of points
    x_min = int(math.floor(pts[:,0].min()))
    x_max = int(math.ceil(pts[:,0].max()))
    y_min = int(math.floor(pts[:,1].min()))
    y_max = int(math.ceil(pts[:,1].max()))

    return(x_min, x_max, y_min, y_max)


def crop_image_by_boundary(image, x_lower, x_upper, y_lower, y_upper, resize=None):
    # crops and optionally resize image based on bounding box
    image_out = image[y_lower:y_upper, x_lower:x_upper]
    if resize:
        image_out = cv2.resize(image_out, resize, interpolation = cv2.INTER_AREA)
    return image_out


def dbscan_crop(img0, img1, mkpts0, mkpts1, min_samples=5, eps=30):
    # initialize return object
    res = {}

    # determine which set has the largest coverage
    cluster0 = DBSCAN(eps=eps, min_samples=min_samples).fit(mkpts0)
    coverage0 = (len(cluster0.labels_) - np.count_nonzero(cluster0.labels_))/len(cluster0.labels_)
    cluster1 = DBSCAN(eps=eps, min_samples=min_samples).fit(mkpts1)
    coverage1 = (len(cluster1.labels_) - np.count_nonzero(cluster1.labels_))/len(cluster1.labels_)

    # combine all point-pairs into dataframe for joining
    pts = pd.DataFrame()
    pts['x0'] = mkpts0[:,0]
    pts['y0'] = mkpts0[:,1]
    pts['x1'] = mkpts1[:,0]
    pts['y1'] = mkpts1[:,1]

    # keep the largest cluster from either image 0 or image 1 via joining
    cluster = pd.DataFrame()
    if coverage0 > coverage1:
        cluster['x0'] = cluster0.components_[:,0]
        cluster['y0'] = cluster0.components_[:,1]
        pd.merge(pts, cluster, on=['x0','y0'], how='inner')
    else:
        cluster['x1'] = cluster1.components_[:,0]
        cluster['y1'] = cluster1.components_[:,1]
        pd.merge(pts, cluster, on=['x1','y1'], how='inner')

    # separate clusters by image to crop
    cluster0 = pts[['x0','y0']].to_numpy()
    cluster1 = pts[['x1','y1']].to_numpy()

    # turn off pyplot interactive
    plt.ioff()

    # process image0
    x0, x1, y0, y1 = find_xy_boundary(cluster0)
    img0_crop = crop_image_by_boundary(img0, x0, x1, y0, y1)
    # Create a Rectangle patch
    rect0 = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor='r', facecolor='none')

    # process image1
    x0, x1, y0, y1 = find_xy_boundary(cluster1)
    img1_crop = crop_image_by_boundary(img1, x0, x1,  y0, y1)
    # Create a Rectangle patch
    rect1 = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor='r', facecolor='none')

    # Create figure and axes
    fig, ax = plt.subplots(2,2)

    # plot
    ax[0,0].imshow(img0, cmap='gray')       # image
    ax[0,0].add_patch(rect0)                    # rect
    x = cluster0[:,0]
    y = cluster0[:,1]
    s = [1 for x in x]
    ax[0,0].scatter(x, y, s)                    # mkpts
    ax[1,0].imshow(img0_crop, cmap='gray')   # image

    ax[0,1].imshow(img1, cmap='gray')       # image
    ax[0,1].add_patch(rect1)                    # rect
    x = cluster1[:,0]
    y = cluster1[:,1]
    s = [1 for x in x]
    ax[0,1].scatter(x, y, s)                    # mkpts
    ax[1,1].imshow(img1_crop, cmap='gray')   # image

    for idx, pt in pts.iterrows():
        con = ConnectionPatch(xyA=(pt.x0,pt.y0), coordsA=ax[0,0].transData,
                              xyB=(pt.x1,pt.y1), coordsB=ax[0,1].transData)
        con.set_alpha(0.1)
        con.set_color('b')
        fig.add_artist(con)

    res['img0_crop'] = img0_crop    # image cropped by dbscan
    res['img1_crop'] = img1_crop
    res['figure'] = fig             # output figure

    # turn on pyplot interactive
    plt.ion()
    return res
