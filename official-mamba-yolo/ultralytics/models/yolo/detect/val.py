# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.extra_metric_keys = ()
        self._visdrone_coco_enabled = False
        self._visdrone_coco_image_ids = {}
        self._visdrone_coco_images = []
        self._visdrone_coco_annotations = []
        self._visdrone_coco_predictions = []
        self._visdrone_coco_ann_id = 1
        self._visdrone_coco_results = None
        self._visdrone_coco_per_class = []

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        if self.args.temporal and "temporal_imgs" in batch:
            batch["temporal_imgs"] = batch["temporal_imgs"].to(self.device, non_blocking=True)
            batch["temporal_imgs"] = (
                batch["temporal_imgs"].half() if self.args.half else batch["temporal_imgs"].float()
            ) / 255
            batch["temporal_valid"] = batch["temporal_valid"].to(self.device, non_blocking=True).float()
        if self.args.temporal and "prev_img" in batch:
            batch["prev_img"] = batch["prev_img"].to(self.device, non_blocking=True)
            batch["prev_img"] = (batch["prev_img"].half() if self.args.half else batch["prev_img"].float()) / 255
            batch["has_prev"] = batch["has_prev"].to(self.device, non_blocking=True).float()
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self._visdrone_coco_enabled = bool(getattr(self.args, "visdrone_vid_coco_metrics", False)) and (
            "VisDroneVID" in str(getattr(self.args, "data", "")) or "VisDroneVID" in str(val)
        )
        if self._visdrone_coco_enabled:
            self.extra_metric_keys = ("metrics/AP(B)", "metrics/AP50(B)", "metrics/AP75(B)")
            self._visdrone_coco_image_ids = {}
            self._visdrone_coco_images = []
            self._visdrone_coco_annotations = []
            self._visdrone_coco_predictions = []
            self._visdrone_coco_ann_id = 1
            self._visdrone_coco_results = None
            self._visdrone_coco_per_class = []
        else:
            self.extra_metric_keys = ()
            self._visdrone_coco_per_class = []

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def _ensure_visdrone_coco_image(self, image_path, pbatch, cls, bbox):
        """Register one validation image and its GT boxes for COCO-style VisDrone metrics."""
        stem = Path(image_path).stem
        image_id = self._visdrone_coco_image_ids.get(stem)
        if image_id is not None:
            return image_id

        image_id = len(self._visdrone_coco_image_ids) + 1
        self._visdrone_coco_image_ids[stem] = image_id
        height, width = int(pbatch["ori_shape"][0]), int(pbatch["ori_shape"][1])
        self._visdrone_coco_images.append(
            {"id": image_id, "file_name": Path(image_path).name, "width": width, "height": height}
        )
        for gt_cls, gt_box in zip(cls.cpu().numpy(), bbox.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in gt_box]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            self._visdrone_coco_annotations.append(
                {
                    "id": self._visdrone_coco_ann_id,
                    "image_id": image_id,
                    "category_id": int(gt_cls) + 1,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            self._visdrone_coco_ann_id += 1
        return image_id

    def _compute_visdrone_coco_metrics(self):
        """Compute COCO-style AP/AP50/AP75 for VisDrone-VID validation."""
        # VisDrone-VID papers commonly report COCO-style AP/AP50/AP75, while
        # the default Ultralytics logs focus on P/R/mAP50/mAP50-95. This
        # adapter exports the current validation set into temporary COCO JSONs
        # and runs pycocotools so the project can report both metric families.
        categories = [{"id": i + 1, "name": self.names[i]} for i in range(self.nc)]
        gt_payload = {
            "images": self._visdrone_coco_images,
            "annotations": self._visdrone_coco_annotations,
            "categories": categories,
        }
        gt_path = self.save_dir / "visdrone_vid_val_gt_coco_tmp.json"
        pred_path = self.save_dir / "visdrone_vid_val_predictions_tmp.json"
        gt_path.write_text(json.dumps(gt_payload, ensure_ascii=False), encoding="utf-8")
        pred_path.write_text(json.dumps(self._visdrone_coco_predictions, ensure_ascii=False), encoding="utf-8")

        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path)) if self._visdrone_coco_predictions else coco_gt.loadRes([])
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = sorted(img["id"] for img in self._visdrone_coco_images)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precision = coco_eval.eval["precision"]  # [TxRxKxAxM]
        area_all_idx = 0
        max_det_idx = 2
        per_class = []
        for class_idx in range(self.nc):
            # The precision tensor layout is [IoU, Recall, Class, Area, MaxDet].
            # Index 0 corresponds to IoU=0.50 and index 5 to IoU=0.75 in the
            # standard COCO threshold grid {0.50, 0.55, ..., 0.95}.
            ap = precision[:, :, class_idx, area_all_idx, max_det_idx]
            ap = ap[ap > -1]
            ap50 = precision[0, :, class_idx, area_all_idx, max_det_idx]
            ap50 = ap50[ap50 > -1]
            ap75 = precision[5, :, class_idx, area_all_idx, max_det_idx]
            ap75 = ap75[ap75 > -1]
            per_class.append(
                {
                    "class_id": class_idx,
                    "class_name": self.names[class_idx],
                    "AP": float(ap.mean()) if ap.size else float("nan"),
                    "AP50": float(ap50.mean()) if ap50.size else float("nan"),
                    "AP75": float(ap75.mean()) if ap75.size else float("nan"),
                }
            )
        self._visdrone_coco_results = {
            "metrics/AP(B)": float(coco_eval.stats[0]),
            "metrics/AP50(B)": float(coco_eval.stats[1]),
            "metrics/AP75(B)": float(coco_eval.stats[2]),
        }
        self._visdrone_coco_per_class = per_class
        per_class_path = self.save_dir / "visdrone_vid_val_per_class_metrics.json"
        per_class_path.write_text(json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8")
        return self._visdrone_coco_results

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            image_id = None
            if self._visdrone_coco_enabled:
                image_id = self._ensure_visdrone_coco_image(batch["im_file"][si], pbatch, cls, bbox)
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            if self._visdrone_coco_enabled and image_id is not None:
                for det in predn.cpu().numpy():
                    x1, y1, x2, y2, score, pred_cls = det.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0 or h <= 0:
                        continue
                    self._visdrone_coco_predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(pred_cls) + 1,
                            "bbox": [x1, y1, w, h],
                            "score": float(score),
                        }
                    )

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        results = self.metrics.results_dict
        if self._visdrone_coco_enabled:
            results = {**results, **(self._visdrone_coco_results or self._compute_visdrone_coco_metrics())}
        return results

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self._visdrone_coco_enabled:
            coco_metrics = self._visdrone_coco_results or self._compute_visdrone_coco_metrics()
            LOGGER.info(
                "VisDrone-VID COCO metrics: "
                f"AP={coco_metrics['metrics/AP(B)']:.4f}, "
                f"AP50={coco_metrics['metrics/AP50(B)']:.4f}, "
                f"AP75={coco_metrics['metrics/AP75(B)']:.4f}"
            )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )
        if self._visdrone_coco_enabled and self.nc > 1 and len(self.stats):
            det_by_class = {
                int(c): self.metrics.class_result(i) for i, c in enumerate(self.metrics.ap_class_index)
            }
            LOGGER.info(
                "%22s%11s%11s%11s%11s%11s%11s%11s%11s%11s"
                % ("Class", "P", "R", "mAP50", "mAP95", "AP", "AP50", "AP75", "Images", "Instances")
            )
            for row in self._visdrone_coco_per_class:
                class_id = int(row["class_id"])
                p, r, map50, map95 = det_by_class.get(class_id, (float("nan"),) * 4)
                LOGGER.info(
                    "%22s%11.4f%11.4f%11.4f%11.4f%11.4f%11.4f%11.4f%11d%11d"
                    % (
                        row["class_name"],
                        p,
                        r,
                        map50,
                        map95,
                        row["AP"],
                        row["AP50"],
                        row["AP75"],
                        int(self.nt_per_image[class_id]) if self.nt_per_image is not None else 0,
                        int(self.nt_per_class[class_id]) if self.nt_per_class is not None else 0,
                    )
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
