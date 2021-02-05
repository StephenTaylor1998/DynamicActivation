python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/resnet18-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a resnet34 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/resnet34-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/resnet50-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a resnet101 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/resnet101-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a resnet152 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/resnet152-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b0 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb0-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b1 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb1-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b2 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb2-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b3 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb3-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b4 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb4-train90-lr0.1-batch1024-TinyImageNet200.txt && \
python main.py -a b5 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' --multiprocessing-distributed\
 --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/TinyImageNet200/ -c 200 -b 1024 -j 20\
 >> ./data/logs/efficientnetb5-train90-lr0.1-batch1024-TinyImageNet200.txt



