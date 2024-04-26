import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import binary_dilation, disk
def draw_poly_mask(poly_coords, im_shape, outline=0):
    #poly_coords: (N,2)
    image = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    poly_coords =poly_coords.flatten()
    d.polygon(poly_coords.tolist(), fill=1, outline=outline)
    return np.array(image)
def boundaryF_from_poly(poly1,poly2,h,w):
    # poly1: (N,2)
    # poly2: (N,2)
    # h,w: int
    # return: F, P, R, fg_match, n_fg, gt_match, n_gt
    poly1_mask=draw_poly_mask(poly1,(h,w),outline=1)
    poly2_mask=draw_poly_mask(poly2,(h,w),outline=1)
    return db_eval_boundary(poly1_mask,poly2_mask)

""" F Boundary Functions

    Taken directly from
    https://github.com/fperazzi/davis-2017/blob/
        7f9de7aff28fee75318b054b90ef0c552b0bb252/python/
        lib/davis/measures/f_boundary.py
"""
def db_eval_boundary(foreground_mask, gt_mask, bound_th=2):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall)

    return F, precision, recall#, np.sum(fg_match), n_fg, np.sum(gt_match), n_gt


def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """
    seg = seg.astype(bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1

    return bmap