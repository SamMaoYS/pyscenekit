import os
import numpy as np
import cv2
import sys
import random

from vp_estimation_with_prior_gravity.features.line_detector import LineDetector
from vp_estimation_with_prior_gravity.solvers import run_hybrid_uncalibrated
from vp_estimation_with_prior_gravity.evaluation import (
    project_vp_to_image,
    get_labels_from_vp,
)
from vp_estimation_with_prior_gravity.visualization import (
    plot_images,
    plot_vp,
    plot_lines,
)


class VPEstimationPriorGravity:
    def __init__(self):
        self.SOLVER_FLAGS = [True, True, True, True, True]

        self.th_pixels = 3  # RANSAC inlier threshold
        self.ls_refinement = 2  # 3 uses the gravity in the LS refinement, 2 does not. Here we use a prior on the gravity, so use 2
        self.nms = 1  # change to 3 to add a Ceres optimization after the non minimal solver (slower)
        self.magsac_scoring = True

        self.gravity = np.array([0.0, 1.0, 0.0])
        self.line_type = "deeplsd"

    def set_gravity(self, gravity: np.ndarray = np.array([0.0, 1.0, 0.0])):
        # We use a prior vertical gravity
        if (self.SOLVER_FLAGS == [True, False, False, False, False]) or (
            self.SOLVER_FLAGS == [False, False, True, False, False]
        ):
            self.gravity = np.array([random.random() / 1e12, 1, random.random() / 1e12])
            self.gravity /= np.linalg.norm(self.gravity)
        else:
            self.gravity = gravity

    def set_line_detector(self, line_type: str = "deeplsd"):
        self.line_type = line_type

    def detect_lines(self, image: np.ndarray):
        line_detector = LineDetector(line_detector=self.line_type)
        lines = line_detector.detect_lines(image)[:, :, [1, 0]]
        return lines

    def estimate_vps(self, image: np.ndarray, visualize: bool = True):
        lines = self.detect_lines(image)
        principle_point = np.array([image.shape[1] / 2, image.shape[0] / 2])
        f, vp = run_hybrid_uncalibrated(
            lines - principle_point[None, None, :],
            self.gravity,
            th_pixels=self.th_pixels,
            ls_refinement=self.ls_refinement,
            nms=self.nms,
            magsac_scoring=self.magsac_scoring,
            sprt=True,
            solver_flags=self.SOLVER_FLAGS,
        )
        vp[:, 1] *= -1

        if visualize:
            K = np.array(
                [[f, 0, principle_point[0]], [0, f, principle_point[1]], [0, 0, 1]]
            )
            vp_labels = get_labels_from_vp(
                lines[:, :, [1, 0]],
                project_vp_to_image(vp, K),
                threshold=self.th_pixels,
            )[0]

            plot_images([image, image])
            plot_lines([lines, np.empty((0, 2, 2))])
            plot_vp([np.empty((0, 2, 2)), lines], [[], vp_labels])
            from matplotlib import pyplot as plt

            plt.savefig("vp_estimation.png")

        return f, vp
