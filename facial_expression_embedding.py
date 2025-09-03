import av, os, cv2, numpy as np, torch, math
import json
from fractions import Fraction
from collections import Counter
from openface.multitask_model import MultitaskPredictor
from subprocess import run
import facial_expression as fe

DEVICE = fe.DEVICE
VIDEO_DIR = "/mnt/data1/videos"
BBOX_DIR = "/mnt/data1/asd_bbox_per_sec_jsons"
ASD_DIR = "/mnt/data1/asd_orch_jsons"

def main():
