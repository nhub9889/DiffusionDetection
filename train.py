import torch
from src.core.model import DiffusionDetModel
from src.data.dataloader import DataLoader
from src.data.coco.coco_dataset import CocoDetection, ConvertCocoPolysToMask

import argparse
model = DiffusionDetModel("resnet18.yaml")
