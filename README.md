* [practical reinforcement learning](https://www.coursera.org/learn/practical-rl/)
* [bayesian-methods-in-machine-learning](https://www.coursera.org/learn/bayesian-methods-in-machine-learning)



## setup
```bash
docker build -t courser:latest .
nvidia-docker run -it -v $PWD:/code/ --net=host --ipc=host courser:latest /bin/bash
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```
