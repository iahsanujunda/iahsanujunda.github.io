---
title: Coffee Analytics 1 - How to Tell if Changing Grind Size Results in Better Brew
date: 2022-08-04 21:53:09
tags:
mathjax: true
---
 
 > TL;DR the smaller grind size behaves pretty much the same with larger grind size. There might be other more important factor at play, or there are too little observation to make a confident conclusion.

It's now been two years since I seriously started making my own coffee. COVID-19 pandemic really brings out my inner barista. Before the pandemic and the ensuing restriction hits, I always buy my coffee. But now I weigh my coffee, hand grind beans right before brewing, 3D print custom tools to help me brew better, and watched [James Hoffmann](https://www.youtube.com/channel/UCMb0O2CdPBNi-QqPk5T3gsQ) videos to know the latest poison to buy.
 
When I want to brew something for both me and my wife, I usually go with a moka pot, because it is one that both of us will always enjoy. I can brew v60 but my wife does not always like the taste of v60 coffee, because my v60 brewing is inconsistent. V60 is a brewer that requires dedication. There are a lot more factors that are involved in making v60 coffee compared to moka pot, so it is really easy to introduce unintended factors into my v60 brewing.
 
![a moka pot](https://drive.google.com/uc?export=view&id=1jvFRi9RrL2jQ9WPFh4QHgsN8fS3R1KBq)
 
With the moka pot, there are much less moving parts that I need to manually control, so it is easier to make a consistent coffee. I just have to use a standard set of configurations that both of us mostly enjoy. But a problem arose when one day I changed one configuration, a smaller grind size, in order to have a stronger cup. When I first made it, it was okay, the cup is indeed a bit more bitter but also more flavorful. But when my wife makes it with the new configuration, it results in a coffee that has barely any foam and it is not as bitter as it should. At first I discarded it as her mistaking a different config, but then I also managed to get a similar result with hers, albeit rarely.
 
This piqued my curiosity. Does the finer grind make a more inconsistent brew? If so, how should I debug this.
 
<!-- more -->
 
## The premise
 
Before I get into analysing anything, I should learn about some missing particulars of my coffee-brewing, to make sure that the data and therefore any potential conclusions I make is not misguided by my own ignorance.
 
Many people take the statement that using grinders, any grinder, result in better tasting coffee. The theory is sound, coffee is hygroscopic [1], meaning it absorbs water from the air, and with it also comes particles and odours. In the bean form, the things we like in a coffee are better protected. The moment the beans are ground, it releases all the coffee goodness into nothingness. By this logic, the shorter the gap between grinding our beans to brewing them, the better[2]. If I pre-ground my coffee from a place that can replicate my coffee grinding levels with a really great grinder, I might not observe the proper effect due to the fact that a lot more coffee goodness has evaporated.
 
Next I have to know how a grind size might affect resulting coffee. Quick research allows me to understand that the grind size affects contact time of water with the coffee. Finer ground coffee will let water seep through slower than a coarser ground, practically 'cooking' them longer. As with cooking anything, we don't want to overcook, because things that are overcooked taste unpleasant. When I change my grind size into a smaller one, I increase brewing time and might potentially overbrew them. But sometimes I got good results. Sometimes the resulting coffee is great. So I need to research some more.
 
The next research enlightens me on particle uniformity. A uniform ground coffee will result in better tasting coffee because the surface area of each particle that is exposed to water is similar. Imagine boiling two potatoes, one potato having twice the size of the other one. More portion of the smaller potato will be in contact with boiling water and therefore will cook faster than the bigger one. The same principle applies with coffee, if our grinder can not grind uniformly, we will have particles with sizes all over the chart. When we brew these, the smaller particles will 'cook' faster, and anything will taste bitter when overcooked. The bigger ones, on the other hand, will be 'undercooked' and will taste weak. Our tongue won't like the combination of overcooked bitterness and undercooked weakness. This might be what happens with me, my grinder loses ability to grind uniformly in lower configuration.
 
## The experiment
 
With necessary knowledge obtained, I now have better understanding on what to suspect, how to prove this suspicion, and therefore getting closer to a consistently good tasting coffee.
 
The first factor that I will put into my analysis is, of course, grind size.
 
![my grinder setting](https://drive.google.com/uc?export=view&id=1ai2dU-VSN9RHlxlYR0CZDrXR6G6Xf4Ng)
 
This is how to set my grinder setting. I just turn this knob clockwise for finer grind size, counter-clockwise for coarser size. There is a kind of stopper mechanism that will click when I rotate this knob, therefore I usually count the number of clicks to adjust my grind size. If I turn the knob clockwise until it can not move anymore, it is the 0 click position. Then I turn the knob counterclockwise until I hear the first click, it is the 1 click position. My previous configuration is using 12 clicks grind size, the new one is 11 clicks. I will use this as the independent variable.
 
![foam](https://drive.google.com/uc?export=view&id=19U99HV3nJPLmHbszx-59o7cLavQdXLJj)
 
Now how should we measure the effect in our coffee? In my daily brewing, I use one proxy metric: foam quantity. I always associate moka pot coffee with more creamy foam on top as better than one without foam. I assumed that more foam means more flavour extracted. But as I researched about this topic, I understand that it is not always the case. The foam in the moka pot is the result of CO2 trapped inside the beans that gets released during brewing. Because it is gas, it will gradually evaporate into the air instead of getting incorporated into the coffee liquid. We can stir the foam to mix with the coffee liquid, but it will not bind better compared to if it is bound with water in the first place. The coffee liquid goodness comes from water, so we should prefer more water rather than more foam.
 
To have a more accurate analysis, I should measure total dissolved particles in my cup of coffee. Professional baristas use a refractometer as part of their routine. Refractometer measures the density of dissolved particles in a water. Better tasting coffee usually has more sugar extracted per unit mass of water. However I don't currently have access to refractometer, and getting one to satisfy my idle curiosity seems overkill, therefore I am back with measuring foam quantity. My justification is that without access to proper measuring tools, we can focus on what comes out instead of how the taste is like.
 
Now we have independent and response variables sorted, we can start collecting data.
 
![forms](https://drive.google.com/uc?export=view&id=1Hg-57AbqcmTc8KPm-PvDH44r2riILKH2)
 
I created a google form to easily organise my data collection. The response I put into this form is conveniently stored in spreadsheet format, so I import them easily when I am ready for analysis.
 
Once I have made several batches of coffee, I revisit my spreadsheet and prepare it for analysis.
 
![spread sheet](https://drive.google.com/uc?export=view&id=1MfTTDjhwr3h1jTtymwBSq5SR2IqksLS1)
 
One thing I did is to create a copy of the working sheet. I might want to reuse this form for more experiments, so it is better to work on the copy of the sheet. I then renamed the headers so that I can process them easier in python. I also need to transform the photo I took of the foam into two categories: low amount of foams or high amount of foams. I simply take a look at each image url, and decide if it is the photo of a low amount or high amount.
 
![high amount](https://drive.google.com/uc?export=view&id=1OlqX7isPUUKIGVruJ3fjjfzSjPztzOj3)
<center>this is high</center>
 
![low amount](https://drive.google.com/uc?export=view&id=1F3MsH5vTaWivFZ-1PiMY6R0d01IbOjKJ)
<center>this is low</center>
 
> The column for foam quantity is called crema_level here. I know it is semantically wrong, but it is a term that a lot of people are more familiar with.
 
Now that I have nicely formatted tabular data. I can easily use the spreadsheet in python for analysis. The python can be easily accessed here: https://colab.research.google.com/drive/1IjGnfKwTFgcCaXcBnfFd0Mn27bx-gYtg?usp=sharing
 
The data we are dealing with can be presented as a 2x2 contingency table.
 
![grind vs crema](https://drive.google.com/uc?export=view&id=1lG4-1Y-qnK154S7WGCihaZ4BqRvBNOY0)
 
It is tempting to use chi square test for independence for this kind of problem because it test whether or not two variables are independent with each other, if `crema_level` is dependent with `grind_level` then changing `grind_level` will gave us different `crema_level`. However we can see that it has a low number of observations. Chi square test for independence will give us a p value result that is lower than if we run the same experiment with more observation [3]. To get around this problem, we can use fisher exact test. With the exact test, instead of testing if `grind_level` is independent with `crema_level`, we are testing if the probability of getting high `crema_level` is the same when we use `grind_level` 11 and 12.

```
H_{0}: probability of getting high crema_level is the same with grind_level 11 and 12.
H_{1}: probability of getting high crema_level is NOT the same with grind_level 11 and 12.
```
 
Running our observation through `scipy.stats.fischer_exact`, we get a p value of 1.0.
 
## Conclusion and Further Work
 
We got a very high p value that tells us that if the probability of getting high `crema_level` is the same with grind size 11 and 12, we are going to see results similar to our observation most of the time. So we accept the hypothesis that when changing grind size from 12 to 11, no change in probability of getting high `crema_level`.
 
Preparing for this experiment has given me a better understanding of coffee-brewing. As I collect info and data, I realised some of my initial understanding is flawed, therefore I have to repeat this experiment with improved factors, constants, variables, and treatments. In the next experiment, I could put my coffee bean in individual zipper bags when it was first bought. I currently put my coffee beans in the bag I bought them in. When I do this, the beans that are used in later batches are exposed more to the air, and therefore lose more of their CO2 and other coffee goodness. This might be what hinders our observation.  If I put them in separate bags that contain only enough for one serving, I can keep the bean in a vacuum longer and only expose them to air once I am about to grind them.
 
As a fun exercise, I ran the same analysis to see the effect of `fire_level` and `crema_level`, the intuition is that `fire_level` affects how fast water vapour seeps through the coffee bed. Finer grind setting might not be overcooked if the cooking time is fast. The result is surprisingly the same, p value is `1.0`.
 
## Reference
 
[1] https://coffeeaffection.com/coffee-in-freezer/
[2] https://youtu.be/bgjvLQu5NlE?t=128
[3] http://www.biostathandbook.com/small.html
[4] http://www.biostathandbook.com/fishers.html

