#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed \
#--world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/

#python main.py -a densenet161 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
# --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/ -c 10 -b 512 -j 20 \
# >> ./data/logs/densenet161-train90-lr0.1-batch1024.txt

#python main.py -a b0 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
# --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/ -c 1000 -b 512 -j 20 \
# >> ./data/logs/efficientnetb0-train90-lr0.1-batch1024.txt

python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8890' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1000 -j 30 --epochs 1\
 >> ./data/logs/efficientnetb0-train90-lr0.1-batch1024.txt
#python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:8890' --dist-backend 'nccl' --multiprocessing-distributed\
# --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 5120 -j 30 \
