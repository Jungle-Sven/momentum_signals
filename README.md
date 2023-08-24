A tool that analyzes tick data from crypto exchanges to produce trade signals.
Based on imbalance in executed trades and orderbook skew. 

1 build

sudo docker build -t momentum_signals .

2 run

sudo docker run -d --restart always --name momentum_signals_container --network momentum_network --log-opt mode=non-blocking --log-opt max-size=10m --log-opt max-file=3 momentum_signals

3 stop

sudo docker stop momentum_signals_container
    
4 remove

sudo docker rm momentum_signals_container

5 logs

sudo docker logs momentum_signals_container
