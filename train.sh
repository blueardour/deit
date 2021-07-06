
#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

#python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../data/imagenet

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp

#python main.py --eval --resume ~/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth --model deit_ms_tiny_patch16_224 --data-path ../../data/imagenet

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/imagenet --output_dir exp --ms_policy config/policy_tiny-8bit.txt

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp/deit_ms_tiny_patch16_224-cifar-bs256-baseline --num_workers 10 --ms_policy config/policy_tiny-baseline.txt

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp/deit_ms_tiny_patch16_224-cifar-bs256-FC16 --num_workers 10 --ms_policy config/policy_tiny-16bit.txt

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp/deit_ms_tiny_patch16_224-cifar-bs256-ALL16 --num_workers 10 --ms_policy config/policy_tiny-16bit.txt

#python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp/deit_ms_tiny_patch16_224-cifar-bs256-ALL8 --num_workers 10 --ms_policy config/policy_tiny-8bit.txt

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_ms_tiny_patch16_224 --batch-size 256 --data-path ../../data/cifar100 --data-set CIFAR --input-size 224 --output_dir ./exp/deit_ms_tiny_patch16_224-cifar-bs256-ALL8-stable1 --num_workers 10 --ms_policy config/policy_tiny-8bit.txt
