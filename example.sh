
#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

#python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../data/imagenet

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

#python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_ms_tiny_patch16_224 --data-path ../../data/imagenet

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp --ms_policy config/policy_tiny-8bit.txt

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp --num_workers 10 --ms_policy config/policy_tiny-8bit.txt

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input_size 32 --output_dir ./exp --num_workers 10 --ms_policy config/policy_tiny-8bit.txt
