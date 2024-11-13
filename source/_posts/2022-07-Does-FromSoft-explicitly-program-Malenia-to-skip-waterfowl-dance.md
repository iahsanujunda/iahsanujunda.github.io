---
title: Does FromSoft explicitly program Malenia to skip waterfowl dance?
date: 2022-07-30 21:05:32
tags: statistics
mathjax: true
---

Hidetaka Miyazaki and Fromsoft has a portfolio of hard and challenging bosses in their games. However, in their latest game Elden Ring, one particular boss has been making a scene online, so much so that videos of her have popped up on my Youtube recommendation before I even purchase the game. She heals with every connecting hit, her move set is unforgiving, she attacks as soon as we try to heal, and one particular move of her is near-impossible to dodge or block. The only way to avoid this attack is to `git gud`!

![Waterfowl Dance](/images/2022-07-Does-FromSoft-explicitly-program-Malenia-to-skip-waterfowl-dance/waterfowl-dance.gif)

This boss is called Malenia. Her and that one particular move, called waterfowl dance, gave PTSD to a lot of people. I would recommend you to search for clips of people fighting Malenia to actually see how formidable she is, but you can watch [this video](https://youtu.be/A-1WrCdd6TY?t=285) on youtube that is quite hilarious. You can even recognised waterfowl dance immediately.

I beat Malenia twice, spread out in 297 attempts. Nearing the end, I got quite good at dodging her attacks and getting the timing right for counter attack. During one of the many attempts, I realized that she did not use her signature move at all and that surprised me. In the end, I learned the hard way that getting surprised while fighting Malenia is a sure way to die.

However, that got me thinking, to make Malenia a consistently hard boss, I reckon FromSoft would program her to always do waterfowl dance. In the off chance that she skips it, that would be completely unintentional. But if they intentionally program Malenia to skip the waterfowl dance, I am curious about their motivation. Is it to reward players who fight her time and time again with an easy run in-between? or is it to actually make the fight harder because players would always be on edge to wait for waterfowl dance?

To satisfy this curiosity, I run a statistical analysis on this problem.

<!-- more -->

## The analysis

Like all statistical analysis, we start by working on a hypothesis. If FromSoft _*unintentionally*_ make Malenia skips waterfowl dance, I would assume that the probability would be quite small. To make an assumption on a number that is small enough for my hypothesis, I searched around forums and online discussions to see average of how many times people are fighting Malenia. A lot of people beat her within dozens to 100 attempt range, with some famous outlier who fight her 200 and 3000 times. Based on this info, I would assume FromSoft designed Malenia in a way that can be beaten if a player faced her 100 times on average. So, Malenia skipping waterfowl dance 1 in 100 attempts seem like a reasonably low probability of happening. If we can prove statistically that the probability is higher than this, then it might be an indicator that FromSoft _*intentionally*_ program her to skip it.

<div style="text-align: center;">
$H_{0}: \pi = 0.01$

$H_{1}: \pi > 0.01$
</div>

This problem could be tackled as a one-sample binomial test. This statistical technique usually taught in terms of coin flip fairness in class. But now instead of expecting a fair 50:50 chance of getting head or tail, we are expecting 1:99 chance of seeing Malenia skips waterfowl dance.

To run this test, we need to collect some information, namely:

* `n`: number of times we fight Malenia
* `x`: number of times Malenia skips waterfowl dance
* `p`: hypothesised probability of Malenia skipping waterfowl dance

We had information about `p` but not the rest. To collect those is the main reason I fight Malenia the second time. I trained until I managed to dodge almost all her attacks and stayed alive consistently long enough to see if she would do a waterfowl dance.

I shot for consistently not dying in 4 minutes, if she didn't do waterfowl dance within those 4 minutes then it is counted towards `x`. The reason I choose 4 minutes is also the result of looking up online. Beating Malenia is quite a big deal especially during early days of Elden Ring release, so people have been posting their win over her. I study the approximate average time needed to beat Malenia from those clips. One channel owned by youtuber [KleinTsuboi](https://www.youtube.com/channel/UCDUpJh1Ek3plo34sGriwe-w) has been especially helpful, as they post hours and hours worth of samples. KleinTsuboi themself is a legend among Elden Ring community due to their character aptly named "Let Me Solo Her" has helped thousands of struggling tarnished in fighting -and winning- against Malenia alone.  

I record some of my sample collection run against Malenia on instagram. If you fancy watching me getting floored, [check this out](https://www.instagram.com/stories/highlights/18046330489350533/).

Once I managed to consistently stayed alive for 4 minutes, I collect the sample and observed that Malenia skip waterfowl dance once in 29 fights.

To run the binomial test, I simply use the `scipy.stats.binom_test` function.

```
p_val = scipy.stats.binom_test(x=1, n=29, p=0.01, alternative='greater')
```

We can use the argument `alternative='greater'` when we want to accept alternative hypotheses when our observation is greater than expected probability.

Running this in python got us p value of `0.25`. It basically means, if it is true that FromSoft makes Malenia skips waterfowl dance once in 100 fights, then we get `0.25` probability of observing Malenia skips waterfowl dance once in 29 fights.

## Making conclusion and potential improvement work

Following statistical convention, we can use alpha of `0.05`. Alpha is the threshold of deciding if our observation is unusual, p value less than this alpha can be regarded as unusual.

Our observation is `0.25`. In relatively friendlier human language, we can say that if FromSoft indeed unintentionally makes Malenia skips waterfowl dance, it is NOT UNUSUAL to observe her skip waterfowl dance once in 29 fights. So we can accept the hypothesis that it is not FromSoft's intention to make Malenia skip the waterfowl dance.

Getting back to the reason on why I did this analysis at all, to know whether or not Malenia skipping waterfowl dance is intentional. If it is, I could work on a strategy that will prepare me fighting her without waterfowl dance. Now we know that it is not intentional, I could always prepare myself to always be at the ready to face waterfowl dance and not caught by surprise when she skips it.

We could argue that the probability of Malenia skipping waterfowl dance is not binomial but instead multinomial, seeing as Malenia can do multiple waterfowl dances in a single fight. This will allow us to observe a more balanced probability among various levels of observation compared to using binary observation where the probability will be heavily biased towards Malenia NOT skipping waterfowl dance. 

We could also safeguard our conclusion from type I and type II error by doing power analysis.

Type I could appear when we fight Malenia three times, got her to skip waterfowl dance in the third fight, and win. When we put this observation into `binom_test`, we get p value of `0.03`. All else being equal, we might conclude that FromSoft intentionally make Malenia skips waterfowl dance.

Type II error appear when we use inappropriate assumption. For example, it is perfectly possible that FromSoft intentionally make Malenia skip waterfowl dance in exactly `0.01` probability. Which means our hypothesis is wrong in the first place.

Power analysis shows probability of seeing an effect if there is an effect to observe. Information needed to run power analysis includes effect size and number of trials. We can switch the context of power analysis, if we want to have fixed probability of seeing an effect, we can calculate number of trials to run to get this probability. Doing too few trials and we might fail to observe effect we want to see, and doing too many trials will not contribute much to the final conclusion. 

I say it's enough reading, now go beat some sense into Malenia!
