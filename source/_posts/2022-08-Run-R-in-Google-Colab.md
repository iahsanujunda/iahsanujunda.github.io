---
title: Run R in Google Colab
date: 2022-08-21 00:33:13
tags:
---

Google colab offers a way to easily run R commands. It is really useful for someone like me who simply wants to try R but does not want to download and set-up the tools needed to run R. There are two approaches, one is to create a new notebook with R runtime, the other one is to execute a single cell with R script inside pre-existing python runtime notebook.

## New R Runtime

To create a new notebook with R, simply go to the following url [http://colab.to/r](http://colab.to/r). It will resolve to `https://colab.research.google.com/notebook#create=true&language=r` which is the url for creating google colab notebook with R runtime.

## Run a Cell with R Script

If we already have existing notebook with python runtime, we can execute a single cell containing R using magic command.

```
# activate R magic
%load_ext rpy2.ipython
```

And then on each cell we want to execute R, start 1st line of that cell with `%%R`.

## Reference

https://stackoverflow.com/questions/54595285/how-to-use-r-with-google-colaboratory
