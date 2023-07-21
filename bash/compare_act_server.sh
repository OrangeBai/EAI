for net in vgg16 resnet34; do
  for act in sigmoid tanh relu selu; do
    qsub train.sh --dataset cifar10 --project EAI_comp_act --net ${net} --act ${act} --name ${net}_${act}_bn_init
    qsub train.sh --dataset cifar10 --project EAI_comp_act --net ${net} --act ${act} --name ${net}_${act}_init --no_bn --lr 0.01
    qsub train.sh --dataset cifar10 --project EAI_comp_act --net ${net} --act ${act} --name ${net}_${act}_bn --no_init
    qsub train.sh --dataset cifar10 --project EAI_comp_act --net ${net} --act ${act} --name ${net}_${act} --no_bn --no_init --lr 0.01
    done
done

