---
title: How to Embed Images Hosted on Google Drive
date: 2022-08-01 23:12:02
tags: web-dev
---

Embedding image URL from google drive is actually easy. We just have to convert the URL we got from google drive from this format `https://drive.google.com/file/d/[image_id]/view?usp=sharing` to this format `https://drive.google.com/uc?export=view&id=[image_id]`.

Even easier, just use this tool I make: [gdrive-url-converter](https://gdrive-url-convert.herokuapp.com)

<!-- more -->

I build this tool using React as I have not got a chance to play with React for a long time. It is a good exercise to flex my front-end muscle.

Repository can be found here: [https://github.com/iahsanujunda/gdrive-image-url-converter](https://github.com/iahsanujunda/gdrive-image-url-converter)

### Reference
1. https://dev.to/temmietope/embedding-a-google-drive-image-in-html-3mm9
2. https://codepen.io/DrewJaynes/details/abJNNjb
