# practical-reinforcement-learning
practical reinforcement learning
bash```
docker build -t practical-reinforcement-learning:latest .
nvidia-docker run -it -v %PWD:/code/ --net=host --ipc=host practical-reinforcement-learning:latest /bin/bash
```
