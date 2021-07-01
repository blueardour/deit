
#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

#python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../data/imagenet

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_ms_tiny_patch16_224 --data-path ../../data/imagenet
