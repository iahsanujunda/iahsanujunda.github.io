---
title: Intuition to Recommender System for Implicit Feedback Dataset
date: 2021-05-17 20:59:40
tags:
mathjax: true
---

I have been tinkering with recommender system at work for a few months now. In order to gain deeper understanding on how the model works, how to the training process learns from data, and how to make recommendation from learned model. This post is basically the overview on what I've learnt.

<!-- more -->

This post will rely heavily on paper from Yifan Hu, Yehuda Koren, and Chris Volinsky titled ["Collaborative Filtering with Implicit Feedback Dataset"](http://yifanhu.net/PUB/cf.pdf). The theory laid out in the paper has been incorporated into several open source tools to build recommender systems, most prominently perhaps the [Apache Spark's ALS](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) package.

I would strongly recommended anyone interested in this topic to go check out the paper for a more thorough study.

## User-Item Interaction for Recommendation

In ideal world, our users would inform us their preference to items they have interact with. Imagine a 5 star rating systems in a music streaming app, where every users give their rating based how they are feeling after listening to songs. This is called *explicit data*, seeing as users give their preference *explicitly*.

However, real world is rarely ideal. Sometimes forcing users to give feedback everytime they interact with an item makes for bad experience, or perhaps explicit feedback is simply not possible to collect, for example due to regulation. There are other cases of course where we simply can't collect explicit feedback, and in all those cases we might need to turn to *implicit feedback*.

As the name implies, with implicit feedback data we infer user's preference of an item from their natural interaction with it. It can be number of clicks, number of views, number of purchase, search pattern, or even mouse movements if it can be collected.

With both types of feedback, the rating should give a sense of both *preference* and *confidence*, the former refers to signal of whether or not the user think positively of an item, and the later refers to how strong the feelings are.

Take a sample of 5 star rating, a user might have personal preference cut off in 4 stars, meaning an item starred 4 is positive feedback, while 3 stars or less means negative feedback. Furthermore, within positive range, a 4 indicates weak positive feeling, while a 5 indicates strong positive feeling. This is how preference and confidence is illustrated.

In an implicit spectrum, we could take a look in viewing history of a video from a video streaming service. A video watched multiple times by a user might indicate their positive feeling about that particular video, with the higher number of views indicates a strong positive feeling.

## Matrix Representation of User-Item Interaction

With $u$ number of users and $i$ number of items, we can then build matrix $R$ of size $m\ast n$ that represents this feedback.

<img src="https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/matrix-R.png" alt="matrix R" width="300"/>

Credit: https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

## Going from Matrix Representation to Making Recommendation

In order to make a recommendation, we are going to decompose our original matrix $R$ in order to learn what kind pattern represented by our observed rating. Basically we are going to breakdown our $R$ matrix into several smaller matrices that will represent the pattern of the original $R$.

One popular way to do this is by using technique Singular Value Decomposition (SVD).