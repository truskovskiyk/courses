# practical-reinforcement-learning
practical reinforcement learning

## setup
```bash
docker build -t practical-reinforcement-learning:latest .
nvidia-docker run -it -v $PWD:/code/ --net=host --ipc=host practical-reinforcement-learning:latest /bin/bash
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```
