import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def resolve_workspace_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else WORKSPACE_ROOT / path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/UAVDT.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml', help='model path(s)')
    parser.add_argument('--cfg', type=str, default='', help='training overrides yaml path')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=None, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default=None, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=None, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--optimizer', default=None, help='SGD, Adam, AdamW')
    parser.add_argument('--resume', type=str, default='', help='resume training from checkpoint path')
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', dest='amp', action='store_true', help='enable amp')
    amp_group.add_argument('--no-amp', dest='amp', action='store_false', help='disable amp')
    parser.set_defaults(amp=None)
    val_group = parser.add_mutually_exclusive_group()
    val_group.add_argument('--val', dest='val', action='store_true', help='run validation during training')
    val_group.add_argument('--no-val', dest='val', action='store_false', help='disable validation during training')
    parser.set_defaults(val=None)
    parser.add_argument('--project', default='output_dir/uavdt', help='save to project/name')
    parser.add_argument('--name', default='mambayolo_uavdt', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--auto_eval', action='store_true', help='run formal UAVDT evaluation after training')
    parser.add_argument('--eval_split', default='test', choices=['train', 'val', 'test'], help='dataset split for auto evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size for auto evaluation')
    parser.add_argument('--eval_workers', type=int, default=4, help='workers for auto evaluation')
    parser.add_argument('--eval_output_dir', type=str, default='', help='directory to save auto evaluation outputs')
    opt = parser.parse_args()
    return opt


def run_auto_eval(opt, save_dir: Path, data_path: str) -> None:
    eval_script = WORKSPACE_ROOT / 'scripts' / 'evaluate_uavdt.py'
    if not eval_script.exists():
        raise FileNotFoundError(f'Auto evaluation script not found: {eval_script}')

    best_weights = save_dir / 'weights' / 'best.pt'
    if not best_weights.exists():
        raise FileNotFoundError(f'Best weights not found: {best_weights}')

    if opt.eval_output_dir:
        eval_output_dir = Path(resolve_workspace_path(opt.eval_output_dir))
    else:
        eval_output_dir = WORKSPACE_ROOT / 'output_dir' / 'uavdt_eval' / f'{opt.name}_{opt.eval_split}'

    python_bin = Path(sys.prefix) / 'bin' / 'python'
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    command = [
        str(python_bin),
        str(eval_script),
        '--weights', str(best_weights),
        '--data', data_path,
        '--split', opt.eval_split,
        '--imgsz', str(opt.imgsz),
        '--batch', str(opt.eval_batch_size),
        '--workers', str(opt.eval_workers),
        '--device', str(opt.device),
        '--output-dir', str(eval_output_dir),
    ]

    print(f'Auto evaluation command: {" ".join(command)}', flush=True)
    subprocess.run(command, check=True, cwd=str(WORKSPACE_ROOT))


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    resume_path = resolve_workspace_path(opt.resume) if opt.resume else ''
    args = {
        'data': resolve_path(opt.data),
        'device': opt.device or '0',
        'project': resolve_workspace_path(opt.project),
        'name': opt.name,
    }
    if opt.cfg:
        args['cfg'] = resolve_workspace_path(opt.cfg)
    if opt.epochs is not None:
        args['epochs'] = opt.epochs
    elif not opt.cfg:
        args['epochs'] = 300
    if opt.workers is not None:
        args['workers'] = opt.workers
    elif not opt.cfg:
        args['workers'] = 8
    if opt.batch_size is not None:
        args['batch'] = opt.batch_size
    elif not opt.cfg:
        args['batch'] = 16
    if opt.imgsz is not None:
        args['imgsz'] = opt.imgsz
    elif not opt.cfg:
        args['imgsz'] = 640
    if opt.optimizer is not None:
        args['optimizer'] = opt.optimizer
    elif not opt.cfg:
        args['optimizer'] = 'SGD'
    if opt.amp is not None:
        args['amp'] = opt.amp
    elif not opt.cfg:
        args['amp'] = False
    if opt.val is not None:
        args['val'] = opt.val
    model_conf = resolve_path(opt.config)
    model = YOLO(resume_path if resume_path else model_conf)

    if task == 'train':
        if resume_path:
            model.train(resume=True, **args)
        else:
            model.train(**args)
        if opt.auto_eval:
            run_auto_eval(opt=opt, save_dir=Path(args['project']) / opt.name, data_path=args['data'])
    elif task == 'val':
        model.val(**args)
    elif task == 'test':
        model.val(**args)
    else:
        raise ValueError(f"Unsupported task: {task}")
