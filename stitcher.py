#!/usr/bin/env python3
# Licensed under the MIT License (c) 2017 Kevin Haroldsen

import os
import math
import logging
import argparse
from typing import Union, List, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse.csr as csr
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt


log = logging.getLogger('stitcher')


try:
    feature_finder = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.BFMatcher_create(cv2.NORM_L2)
except AttributeError:
    raise ImportError('You do not have OpenCV SIFT support installed')


def update_defaults(obj, kwargs):
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            raise NameError("Class '%s' does not have an attribute '%s'" % (
                obj.__class__.__name__, k))
        setattr(obj, k, v)


def image_corners(arr):
    return np.array([
        [0., 0.],
        [0., arr.shape[1]],
        arr.shape[:2],
        [arr.shape[0], 0.],
    ])


def fitting_rectangle(*points):
    # Return (left, top), (width, height)
    top = left = float('inf')
    right = bottom = float('-inf')
    for x, y in points:
        if x < left:
            left = x
        if x > right:
            right = x
        if y < top:
            top = y
        if y > bottom:
            bottom = y
    left = int(math.floor(left))
    top = int(math.floor(top))
    width = int(math.ceil(right - left))
    height = int(math.ceil(bottom - top))
    return (left, top), (width, height)


def paste_image(base, img, shift):
    """Fast image paste with transparency support and no bounds-checking"""
    assert base.dtype == np.uint8 and img.dtype == np.uint8
    h, w = img.shape[:2]
    x, y = shift
    dest_slice = np.s_[y:y + h, x:x + w]
    dest = base[dest_slice]
    mask = (255 - img[..., 3])
    assert mask.dtype == np.uint8
    assert mask.shape == dest.shape[:2], (mask.shape, dest.shape[:2])
    dest_bg = cv2.bitwise_and(dest, dest, mask=mask)
    assert dest_bg.dtype == np.uint8
    dest = cv2.add(dest_bg, img)
    base[dest_slice] = dest


def color_stats(lab_image, mask=None):
    """Get color stats of a L*a*b* image (mean, std-dev)"""
    if mask is None:
        # Image is always in 1 dimension to make it simpler
        lab_image = lab_image.reshape(-1, lab_image.shape[-1])
    else:
        lab_image = lab_image[mask.nonzero()]
    return lab_image.mean(axis=0), lab_image.std(axis=0)


def imshow(img, title=None, figsize=None, **kwargs):
    if figsize is None:
        plt.plot()
    else:
        plt.figure(figsize=figsize)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.imshow(img, interpolation='bicubic', **kwargs)
    plt.show()


class _StitchImage:
    _lastIdx = 1

    def __init__(self, image, name: str=None):
        self.image = image
        self.kp = None
        self.feat = None

        if name is None:
            name = '%02d' % (_StitchImage._lastIdx)
            _StitchImage._lastIdx += 1
        self.name = name

    def find_features(self):
        log.debug('Finding features for image %s', self.name)
        self.kp, self.feat = feature_finder.detectAndCompute(self.image, None)


class ImageStitcher:
    def __init__(self, **kwargs):
        self._matches = {}
        self._images = []

        self.ratio_threshold = 0.7
        self.match_threshold = 10
        self._center = None
        self._current_edge_matrix = None
        self.debug = False
        self.correct_colors = False

        update_defaults(self, kwargs)

    def add_image(self, image: Union[str, np.ndarray], name: str=None):
        """Add an image to the current stitching process. Image must be RGB(A)"""
        if isinstance(image, str):
            if name is not None:
                name = os.path.splitext(os.path.split(image)[1])[0]
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGBA)
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image = _StitchImage(image, name=name)
        image.find_features()
        idx = len(self._images)
        self._images.append(image)

        for oidx, other in enumerate(self._images[:-1]):
            match = self._match_features(image, other)
            if match is not None:
                self._matches[(idx, oidx)] = match

    @property
    def center(self) -> int:
        if self._center is None:
            self._center = self._find_center()
        return self._center

    @center.setter
    def center(self, val: int):
        self._center = val

    def stitch(self):
        """Perform the actual stitching - return the image of the result."""
        self.validate()
        log.info('%s considered center image', self._images[self.center].name)
        parents = csgraph.dijkstra(
            self._edge_matrix,
            directed=False, indices=self.center,
            return_predecessors=True,
        )[1]
        log.debug('Parent matrix:\n%s', parents)
        next_H = self._calculate_relative_homographies(parents)
        Hs = self._calculate_total_homographies(parents, next_H)
        all_new_corners = self._calculate_new_corners(Hs)
        base_shift, base_size = np.array(self._calculate_bounds(all_new_corners))
        order = self._calculate_draw_order(parents)
        if self.correct_colors:
            self._correct_colors(parents, next_H, order[1::-1])
        canvas = np.zeros((base_size[1], base_size[0], 4), dtype=np.uint8)
        for i in order:
            image = self._images[i]
            new_corners = all_new_corners[i]
            H = Hs[i]

            shift, size = np.array(fitting_rectangle(*new_corners))
            dest_shift = shift - base_shift
            log.info('Pasting %s @ (%d, %d)', image.name, *dest_shift)

            log.debug('Shifting %s by (%d, %d)', image.name, *shift)
            log.debug('Transformed %s is %dx%d', image.name, *size)
            T = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])
            Ht = T.dot(H)
            log.debug('Translated homography:\n%s', Ht)
            new_image = cv2.warpPerspective(
                image.image, Ht, tuple(size),
                flags=cv2.INTER_LINEAR,
            )
            paste_image(canvas, new_image, dest_shift)
        log.info('Done!')
        return canvas

    def validate(self):
        cc, groups = csgraph.connected_components(self._edge_matrix, directed=False)
        if cc != 1:
            most_common = np.bincount(groups).argmax()
            raise ValueError('Image(s) %s could not be stitched' % ','.join(
                self._images[img].name for img in np.where(groups != most_common)[0]
            ))

    def _correct_colors(self, parents, next_H, order):
        log.debug('Recoloring in order %s', ','.join(self._images[i].name for i in order))
        labs = [cv2.cvtColor(i.image, cv2.COLOR_RGB2LAB).astype(np.float32) for i in self._images]
        for dst_idx in order:
            assert dst_idx != self.center
            src_idx = parents[dst_idx]
            src = self._images[src_idx]
            dst = self._images[dst_idx]
            log.debug('Color Correcting %s => %s', src.name, dst.name)
            src, dst = src.image, dst.image
            src_mask = np.zeros(src.shape[:2], dtype=np.uint8)
            dst_mask = np.zeros(src.shape[:2], dtype=np.uint8)

            H = next_H[src_idx]
            src_corners = cv2.perspectiveTransform(
                image_corners(dst).reshape(1, 4, 2), cv2.invert(H)[1])
            dst_corners = cv2.perspectiveTransform(image_corners(src).reshape(1, 4, 2), H)

            cv2.fillPoly(src_mask, np.rint(src_corners).astype(int), 255)
            cv2.fillPoly(dst_mask, np.rint(dst_corners).astype(int), 255)

            src_lab = labs[src_idx]
            dst_lab = labs[dst_idx]
            # Probably a better way to do this
            # Adapted from http://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
            src_mean, src_std = color_stats(src_lab, mask=src_mask)
            dst_mean, dst_std = color_stats(dst_lab, mask=dst_mask)
            dst_lab[:] = ((dst_lab - dst_mean) * (dst_std / src_std)) + src_mean

        for image, lab in zip(self._images, labs):
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            image.image[:] = cv2.merge(
                cv2.split(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)) +
                [image.image[..., 3]])

    def _calculate_new_corners(self, Hs) -> List[np.array]:
        all_new_corners = []
        for image, H in zip(self._images, Hs):
            corners = image_corners(image.image)
            new_corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2), H)
            if new_corners.shape[0] != 1:
                raise ValueError('Could not calculate bounds for %s!' % image.name)
            new_corners = new_corners[0]
            log.debug(
                '%s transform: (%s,%s,%s,%s)->(%s,%s,%s,%s)',
                image.name, *(
                    '(%s)' % ','.join(str(int(round(i))) for i in arr)
                    for arr in (*corners, *new_corners)),
            )
            all_new_corners.append(new_corners)
        return all_new_corners

    def _calculate_bounds(self, new_corners) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate the bounds required to hold all images transformed with the given corners"""
        all_corners = []
        for corners in new_corners:
            all_corners.extend(corners)
        log.debug('%d new corners to calculate bounds with', len(all_corners))
        corner, size = fitting_rectangle(*all_corners)
        log.info('Center at: %r', (-corner[0], -corner[1]))
        log.info('Final Size: %r', size)
        return corner, size

    def _calculate_draw_order(self, parents):
        order = csgraph.depth_first_order(
            csgraph.reconstruct_path(self._edge_matrix, parents, directed=False),
            self.center,
            return_predecessors=False,
        )[::-1]
        log.info('Draw order: %s', ', '.join(self._images[i].name for i in order))
        return order

    def _calculate_relative_homographies(self, parents):
        # Calculate each homography from the source to the destination
        c = self.center
        next_H = []
        for src_idx, dst_idx in enumerate(parents):
            if dst_idx < 0 or src_idx == c:
                # We are at the center node
                next_H.append(np.identity(3))
                continue
            matches = self._get_match(src_idx, dst_idx)
            swap = (src_idx, dst_idx) not in self._matches
            src, dst = self._images[src_idx], self._images[dst_idx]
            H = self._find_homography(src, dst, matches, swap=swap)
            next_H.append(H)
        return next_H

    def _calculate_total_homographies(self, parents, next_H):
        """Calculate the full homography each picture will have for the final image"""
        # Now that we have the homographies from each to its next-to-center,
        # calculate relative to the center
        c = self.center
        total_H = [None] * len(parents)
        total_H[c] = next_H[c]
        path = []
        while any(i is None for i in total_H):
            path.append(next(n for n, i in enumerate(total_H) if i is None))
            while path:
                src_idx = path.pop()
                dst_idx = parents[src_idx]
                if c == src_idx:
                    continue

                if total_H[dst_idx] is None:
                    # The next node needs to be calculated
                    path.extend((src_idx, dst_idx))
                else:
                    # Matrix multiply src to dst
                    total_H[src_idx] = next_H[src_idx].dot(total_H[dst_idx])
        return total_H

    def _get_match(self, src_idx: int, dst_idx: int):
        if (src_idx, dst_idx) in self._matches:
            return self._matches[(src_idx, dst_idx)]
        return self._matches[(dst_idx, src_idx)]

    def _find_homography(
            self,
            src: _StitchImage,
            dst: _StitchImage,
            matches: List[cv2.DMatch],
            swap=False) -> np.ndarray:
        """Calculate the actual homography for a perspective transform from src to dst"""
        log.info('Transforming %s -> %s', src.name, dst.name)
        if swap:
            src, dst = dst, src
            log.debug('Performing swapped homography find')

        src_data = np.array(
            [src.kp[i.queryIdx].pt for i in matches],
            dtype=np.float64).reshape(-1, 1, 2)

        dst_data = np.array(
            [dst.kp[i.trainIdx].pt for i in matches],
            dtype=np.float64).reshape(-1, 1, 2)

        if swap:
            src_data, dst_data = dst_data, src_data
            src, dst = dst, src

        H, status = cv2.findHomography(src_data, dst_data, cv2.RANSAC, 2.)
        if status.sum() == 0:
            raise ValueError('Critical error finding homography - this should not happen')
        log.debug('Homography for %s->%s:\n%s', src.name, dst.name, H)
        return H

    def _find_center(self) -> int:
        log.debug('Calculating the center image')
        shortest_path = csgraph.shortest_path(
            self._edge_matrix, directed=False,
        )
        log.debug('Shortest path result: %s', shortest_path)
        center = np.argmin(shortest_path.max(axis=1))
        log.debug('The center image is %s (index %d)' % (self._images[center].name, center))
        return center

    @property
    def _edge_matrix(self):
        if len(self._images) == 0:
            raise ValueError('Must have at least one image!')
        current = self._current_edge_matrix
        if current is not None and current.shape[0] == len(self._images):
            return current
        # guarantee same order
        all_matches = list(self._matches)
        base = max(len(v) for v in self._matches.values()) + 1
        # Score connections based on number of "good" matches
        values = [base - len(self._matches[i]) for i in all_matches]
        self._current_edge_matrix = csr.csr_matrix(
            (values, tuple(np.array(all_matches).T)),
            shape=(len(self._images), len(self._images)),
        )
        log.debug('New edge matrix:\n%s', self._current_edge_matrix.toarray())
        return self._current_edge_matrix

    def _match_features(self, src: _StitchImage, dst: _StitchImage) -> Optional[List[cv2.DMatch]]:
        """Match features between two images. Uses a ratio test to filter."""
        log.debug('Matching features of %s and %s', src.name, dst.name)
        matches = matcher.knnMatch(src.feat, dst.feat, k=2)
        # Ratio test
        good = [i for i, j in matches
                if i.distance < self.ratio_threshold * j.distance]
        if self.debug:
            imshow(
                cv2.drawMatches(
                    src.image[..., :3], src.kp,
                    dst.image[..., :3], dst.kp, good, None),
                title='%s matched with %s' % (src.name, dst.name), figsize=(10, 10)
            )
        log.debug('%d features matched, %d of which are good', len(matches), len(good))
        if len(good) >= self.match_threshold:
            log.info('%s <=> %s (score %d)', src.name, dst.name, len(good))
            return good
        return None


def main():
    parser = argparse.ArgumentParser(description='Connects separate images in a panoramic style')
    parser.add_argument(
        'input', nargs='+',
        help='The input image files')
    parser.add_argument(
        '-o', required=True, dest='output',
        help='Where to put the resulting stitched image')
    parser.add_argument(
        '-v', action='count', dest='verbosity', default=0,
        help='Increase the logging verbosity, can be used multiple times')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode, shows intermediate steps for images')
    parser.add_argument(
        '-r', '--ratio-threshold', default=.7, type=float,
        help='The required distance ratio between neighboring feature matches.'
        'Lower means more lenient matching. Default 0.7.')
    parser.add_argument(
        '-b', '--base', type=int,
        help='Specify the index of the base image (that the other images will morph to)')
    parser.add_argument(
        '-m', '--match-threshold', default=10, type=int,
        help='The required number of features matches for two images to be considered "stitchable"')
    parser.add_argument(
        '-c', '--color-correction', action='store_true',
        help='Enable color correction in the resulting output image')
    args = parser.parse_args()

    if args.verbosity > 0:
        log.setLevel(logging.DEBUG if args.verbosity > 1 else logging.INFO)

    stitch = ImageStitcher()
    if args.base is not None:
        stitch.center = args.base
    for infile in args.input:
        stitch.add_image(infile)
    result = stitch.stitch()

    if args.debug:
        imshow(result, title='Final Result')

    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))


if __name__ == '__main__':
    main()
