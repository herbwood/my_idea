import torch
from torch import Tensor


def _box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:

    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes


def _box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes


def _box_xywh_to_xyxy(boxes: Tensor) -> Tensor:

    x, y, w, h = boxes.unbind(-1)
    boxes = torch.stack([x, y, x + w, y + h], dim=-1)
    return boxes


def _box_xyxy_to_xywh(boxes: Tensor) -> Tensor:

    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    boxes = torch.stack((x1, y1, w, h), dim=-1)
    return boxes

if __name__ == "__main__":
    # usage example 
    # xywh -> x1y1x2y2
    boxes = torch.Tensor([[10, 15, 20, 30], [23, 34, 15, 20]])
    transformed = _box_xywh_to_xyxy(boxes)

    """
    tensor([[10., 15., 30., 45.], 
            [23., 34., 38., 54.]])  
    """
    print(transformed)