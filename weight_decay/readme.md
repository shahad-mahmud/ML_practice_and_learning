# Weight decay

Weight decay is one of the widely used regularization techniques. It is also known as L2 regularization. In this technique, we add a penalty term with the loss. Generally, the penalty term is the L2 norm of the weight. Thus if the loss increase, the model might focus on minimizing the L2 norm of the weight. You can find more details [here](https://blog.shahadmahmud.com/weight-decay-basics-with-implementations/).

## Running this experiment

At first clone this repo. Then create a virtual environment and install the required packages from the `requirements.txt` file.
```pip install -r requirements.txt```
Now simply run the `train.ipynb` notebook.

## What is the output?

For the first training, we did not use the weight decay. We notice that the training loss is very small. But during the second training, the training loss is large. Now if we look at the L2 norms of both the training, we see L2 norm of first training is about three times of the second training. This is because the weight decay is used.
