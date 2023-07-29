cd ..
for act in sigmoid tanh relu selu gelu; do
  python train.py --dataset cifar10 --project EAI_comp_preact --net resnet34 --act ${act} --name resnet34_${act} --mode "preact"
  python train.py --dataset cifar10 --project EAI_comp_preact --net vgg16 --act ${act} --name vgg16_${act} --mode "preact"
done