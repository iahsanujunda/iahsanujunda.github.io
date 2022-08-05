---
title: Intuition to Recommender System for Implicit Feedback Dataset
date: 2021-05-17 20:59:40
tags:
mathjax: true
---

I have been tinkering with recommender system at work for a few months now in order to gain deeper understanding on how the model works, how the training process learns from observation data, and how to make recommendation from learned model. This post is basically the overview on what I've learnt and will be divided into several parts, this is the first.

This post will rely heavily on paper from Yifan Hu, Yehuda Koren, and Chris Volinsky titled ["Collaborative Filtering with Implicit Feedback Dataset"](http://yifanhu.net/PUB/cf.pdf). The theory laid out in the paper has been incorporated into several open source tools to build recommender systems, most prominently perhaps the [Apache Spark's ALS](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) package.

<!-- more -->

I would strongly recommended anyone interested in this topic to go check out the paper for a more thorough study.

## User-Item Interaction for Recommendation

In ideal world, our users would inform us their preference to items they have interact with. Imagine a 5 star rating systems in a music streaming service, where every users give their rating based how they are feeling after listening to songs. This is called *explicit data*, seeing as users give their preference *explicitly*.

However, real world is rarely ideal. Sometimes forcing users to give feedback everytime they interact with an item makes for bad experience, or perhaps explicit feedback is simply not possible to collect, for example due to regulation. There are other cases of course where we simply can't collect explicit feedback, and in all those cases we might need to turn to *implicit feedback*.

As the name implies, with implicit feedback data we infer user's preference of an item from their natural interaction with it. It can be number of clicks, number of views, number of purchase, search pattern, or even mouse movements if it can be collected.

With both types of feedback, the rating should give a sense of both *preference* and *confidence*, the former refers to signal of whether or not the user think positively of an item, and the later refers to how strong the feelings are.

Revisiting the 5 star rating example, a user might have personal preference cut off in 4 stars, meaning an item starred 4 is positive feedback, while 3 stars or less means negative feedback. Furthermore, within positive range, a 4 indicates weak positive feeling, while a 5 indicates strong positive feeling. This is how preference and confidence is illustrated.

In an implicit spectrum, we could take a look in viewing history of a video from a video streaming service. A video watched multiple times by a user might indicate their positive feeling about that particular video, with the higher number of views indicates a strong positive feeling.

## Matrix Representation of User-Item Interaction

With $u$ number of users and $i$ number of items, we can then build matrix $R$ of size $u\ast i$ that represents this feedback.

<img src="https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/matrix-R.png" alt="matrix R" width="300"/>

Credit: https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

## Going from Matrix Representation to Making Recommendation

In order to make a recommendation, we are going to decompose our original matrix $R$ in order to learn what kind pattern represented by our observed rating. Basically we are going to breakdown our $R$ matrix into several smaller matrices that will each represent a pattern to build the original $R$.

One popular way to do this is by using technique Singular Value Decomposition (SVD). SVD decomposes our original observation matrix $R$ into three matrices.

<div style="text-align: center;">
$$
R=U\Sigma V^{T}
$$
</div>

where

* $U$ is an orthonormal matrix, whose columns are left singular vector
* $\Sigma$ is a diagonal matrix, whose diagonals are called singular values
* $V$ is an orthonormal matrix, whose columns are right singular vectors

The dimension of each submatrices will be determined based on the original values of $R$, however we can select only top $k$ singular values of each submatrices, therefore the dimension of our submatrices will be:

* $U: u\ast k $
* $\Sigma: k\ast k $
* $V: k\ast i $

Due to the fact that we select only top $k$ singular values, we might lose some information to build the original $R$. Therefore what we get is an *approximation* of $R$ instead, called $\hat{R}$.

<img src="https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/R-SVD.png" alt="R SVD" width="800"/>

Credit: https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

Above illustration shows approximately how SVD looks. Note that the illustration has no $\Sigma$ and the reason is quite simple, when we multiply a matrix with suitably-sized square matrix, it will results in a matrix with the exact same dimension. In this case, $\Sigma$ is the square matrix with a dimension that matches with both $U$ and $V^{T}$, so $\Sigma$ can be absorbed to either one.

We can see that our $U$ and $V$ matrices now contains the pattern from the original user and item interaction. With each row $u$ in $U$ represents a vector that contains the pattern of user $u$. While each column $i$ in $V^{T}$ represents a vector that contains the pattern of item $i$. Because these vectors represent user patterns and item patterns, respectively, in technical terms they are referred to as *latent factor vectors*, with $x_{u}$ is a latent factor vector of user $u$, while $y_{i}$ is a latent factor vector of item $i$. the number of *latent factor* we retain is the number of top $k$ singular values that we select earlier. 

Now that we are able to get latent factors of any particular users and items, making recommendation is simply a process of taking product of $x_{u}$ to $y_{i}$.

<div style="text-align: center;">
$$
\hat{r}_{ui}=x_{u}{y_{i}}^{T}
$$
</div>

With $\hat{r}\_{ui}$ is the recommendation value to give for a particular user and item.

Ideally, when we know the value $r\_{ui}$ , we want $\hat{r}\_{ui}$ to be as close as possible to $r\_{ui}$. The interesting part is where we don't know the actual value of $r\_{ui}$ seeing as our $R$ is sparse. If original $r\_{ui}$ is empty, but $\hat{r}\_{ui}$ return non-empty value, this is the potential recommendation that we discover from our latent factors. With this logic, when we want to recommend items to user $u$, we can dot product all $y_{i}$ to a constant $x_{u}$ to get all $\hat{r}\_{ui}$ for this particular user. We can then sort all items based on the resulting $\hat{r}_{ui}$. The top $\hat{r}\_{ui}$ is the ones we could recommend to the user.
