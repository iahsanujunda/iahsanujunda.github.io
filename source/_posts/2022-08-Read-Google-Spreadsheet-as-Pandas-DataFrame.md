---
title: Read Google Spreadsheet as Pandas DataFrame
date: 2022-08-08 17:26:21
tags: web-app
---

We can read our google spreadsheets via pandas to be used in our analysis. We just have to get this part of the sharing URL `https://docs.google.com/spreadsheets/d/[sheet_id]/edit?usp=sharing` and the sheet name, then put it into this format `https://docs.google.com/spreadsheets/d/[sheet_id]/gviz/tq?tqx=out:csv&sheet=[sheet_id]`.

We can then put this url into `pandas.read_csv()`.

And of course I build a react app for this. [It's this one](http://google-spreadsheet-url-convert.herokuapp.com/).
