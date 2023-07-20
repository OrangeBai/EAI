cd ..

python train.py --dataset cifar10 --net vgg16 --project test --name test --act sigmoid
python train.py --dataset cifar10 --net vgg16 --project test --name test --act tanh
python train.py --dataset cifar10 --net vgg16 --project test --name test --act relu
python train.py --dataset cifar10 --net vgg16 --project test --name test --act selu

python train.py --dataset cifar10 --net vgg16 --project test --name test --act sigmoid --no_bn
python train.py --dataset cifar10 --net vgg16 --project test --name test --act tanh --no_bn
python train.py --dataset cifar10 --net vgg16 --project test --name test --act relu --no_bn
python train.py --dataset cifar10 --net vgg16 --project test --name test --act selu --no_bn

python train.py --dataset cifar10 --net vgg16 --project test --name test --act sigmoid --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act tanh --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act relu --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act selu --no_init

python train.py --dataset cifar10 --net vgg16 --project test --name test --act sigmoid --no_bn --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act tanh --no_bn --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act relu --no_bn --no_init
python train.py --dataset cifar10 --net vgg16 --project test --name test --act selu --no_bn --no_init
