---
title: Starting Analytics Project
date: 2020-06-06 19:05:25
tags:
---

Making a good question is important step of any analytics project, and unfortunately, a concept of bad question exists when it comes to analytics. We are going to identify good ones from bad ones as we go along. I start this project by creating a code book for the dataset that we are working on in this project, it will start from a single variable and it _will_ evolve as we went through all analytics phases.

<!-- more -->

I picked the [gapminder](https://www.gapminder.org/data/) dataset for this project. This dataset hold statistical variables about all UN member countries on social, economic, and environmental development. Among variables in this dataset, I am most interested in analyzing how much CO2 emission produced by each countries.

![gapminder co2 total](https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/gapminder_co2_emission.png)

The simplest form of analytical question is looking for association, so I am going to look for secondary variables that might affect how countries produce CO2 emission. As my mission with DANA Indonesia is bringing internet-powered digital payment service to Indonesian people, I would be curious to see if whether or not my work has positive or negative impact on CO2 emissions. Among gapminder dataset, one of the variable is Rate of internet user (in %)

![gapmnder internet user rate](https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/gapminder_internetuse_rate.png)

To complement this variable, we are going to pull total population as well, in case the number of people is more insightful than the ratio.

![gapminder total population](https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/gapminder_total_population.png) 

Now that we have our two main variables, our research question can be formalized as,

>"Is there an association between CO2 emission produced by each country to the rate of internet users in those countries"

This looks like a reasonable question for an analytics project, as we have the data of both variables that is well defined and we also have clear goal to answer. It is always recommended to have a question that is clear and concise, but we should also need to be careful not to ask a question that is too simple that it would just produce a statement of fact such as, "What is the average internet user for each country?"

Next thing to look for in a good analytical question is that it is not already answered. It would be a waste of time to re-do something that is already been answered before. Doing a literature review would help us in discovering studies done it the past about our problem. A quick copy and pasting our research question through [google scholar](https://scholar.google.com/) would help us discover all previous research done on this topic.

One study by [Lee & Bramahsene (2014)](https://t.umblr.com/redirect?z=https%3A%2F%2Fwww.tandfonline.com%2Fdoi%2Fabs%2F10.1080%2F1226508X.2014.917803&t=OTczNjZhZDY0MmNhYzZjZmFmYWQ4ZDBiMjdmYTZlZDUzODE3YjAxYSxkbnF3MWJIcA%3D%3D&b=t%3AgDEbafbOiheENGoAz6-G3w&p=https%3A%2F%2Fjunda-ia.tumblr.com%2Fpost%2F617662554512179200&m=1) suggested that the correlation between CO2 emission and internet user rate is a positive one. Although in the study they stated that internet user is a confounding factor of economic growth, and economic growth is the one having a direct effect on CO2 emission. Seems like we should add a new variable to our code book to support this hypothesis. We can leverage GDP per capita data on gapminder dataset to represent economic growth. Even though it is not clear at this point how the study done by Lee & Bramahsene accounts for economic growth, for the problem currently at hand, we are going to determine if GDP per capita can be used alond the course of this project.
   
![gapminder gdp per capita](https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/gapminder_gdp_capita.png)

Another study done by [Baliga, et al. (2009)](https://t.umblr.com/redirect?z=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FKerry_Hinton%2Fpublication%2F238634031_Carbon_footprint_of_the_Internet%2Flinks%2F0deec52dc3da3771e5000000%2FCarbon-footprint-of-the-Internet.pdf&t=OWRiYjVhMDU3ZTViMjllMjA2ZGE4OTU5NDVjNGY2ZDNkMmYwNjdjMixkbnF3MWJIcA%3D%3D&b=t%3AgDEbafbOiheENGoAz6-G3w&p=https%3A%2F%2Fjunda-ia.tumblr.com%2Fpost%2F617662554512179200&m=1) mentioned that higher internet usage would encourage fewer business travel, which means a negative correlation between CO2 emission and internet user rate. This paper mentioned that several factor need to be taken into consideration in order for this hypothesis to hold true, first factor is that the quality of service need to be good enough for consistent conferencing usage to replace the need for business travel. This is an interesting angle to look into, and although the gapminder dataset does not contain data about quality of internet, it does contain oil consumption per person that is defined as kilograms of oil consumed per year per person.

![gapminder oil percapita](https://iahsanujunda-hosted-files.s3.us-east-2.amazonaws.com/images/gapminder_oil_percapita.png)

One last interesting hypothesis is made by [Salahuddin, et al. (2016)](https://t.umblr.com/redirect?z=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fabs%2Fpii%2FS1364032116300351&t=MWE1OTM2OTcyMmRlZTdkNDViOTUxZGQ4MWFjYzk5Y2UzY2FjN2FiYyxkbnF3MWJIcA%3D%3D&b=t%3AgDEbafbOiheENGoAz6-G3w&p=https%3A%2F%2Fjunda-ia.tumblr.com%2Fpost%2F617662554512179200&m=1). In their paper, they mentioned that although positive correlation between CO2 consumption and internet usage exists, it is weak as there are more direct and stronger factor that has an impact on CO2 emissions.

Now that we have enough information on previous research done on the topic, we can have a revised question.

> "Is there an association between CO2 emission produced by countries when economic growth and rate of oil consumption is taken into account"

In short summary, where positive/negative correlation exists between internet user rate and CO2 emission, we are going to see if it is due to effect from economic growth and/or oil consumption. This is what we are going to analyze.  
