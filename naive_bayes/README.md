# Naive Bayes

## Quick recaps

Naive Bayes is a probabilistic classifier that uses Bayes' theorem to calculate the probability of a class. Bayes' theorem is stated as:

$$ P(class | data) = \frac{P(class)P(data|class)}{p(data)} $$

It means the probability of a class given some data is the product of the probability of the class and the probability of the data given the class divided by the probability of the data in all classes.

### What is naive in Naive Bayes?

- One crucial assumption Naive Bayes makes, is the independence of features. This means, that the occurrence of one event doesn’t affect the occurrence of the other event. Therefore, all interactions and correlations among the features will simply be ignored.
