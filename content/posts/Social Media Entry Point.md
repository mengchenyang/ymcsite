+++
title = "Social Media Entry Point"
description = "Learn about Mengchen's ongoing HCI research that explores the potential impact of posting scientific paper on Twitter."
tags = [
    "HCI Research","Social Media"
]
date = "2019-09-30"
categories = [
    "",
    "",
]
menu = "main"

+++

## **Introduction**

Scholars tweets about their new paper published or what their research accomplished all the time. Through this Social Media Entry Point project, We would like to find out what exactly people tweet about scientific paper, how people respond to those tweets,  what useful information we could extract from those tweets. By identifying entry point features of tweets, we want to Correlate those features with potential impacts on the scientific paper.

**Project Duration**: 2019/09 - Present   

**Project Supervisor**: Professor [Joel Chan](http://joelchan.me/) @iSchool, University of Maryland

## **How we conduct the research**

First we used Keyword below  to query on the Twitter, filtering the tweets that have scientific contents such as publication of new paper.

| Key Word      |
| ------------- |
| New Paper     |
| Our Paper     |
| Our Reasearch |
| New Research  |
| Our Findings  |
| My paper      |

Having asked a few questions what kind of tweets we are exactly looking for, and literature reviewing papers in the field, 7 features was selected that we believe could be related to social media entry point.

| Features                                  |
| ----------------------------------------- |
| URL of paper                              |
| Include graph/figure?                     |
| Summarize problem/Question?               |
| Summarize Result/Conclusion/Contribution? |
| Threaded?                                 |
| Has replies from others                   |
| From original author                      |

Next I collected around 300 tweets regareding to those features, and quantifyied those data to build a prototype database.

(Below are 3 sample entries.)

| search term / approach | link                                                         | text of "head" tweet                                         | url of paper                  | include graph/figure? | summarize problem/question? | summarize result/conclusion/contribution? | threaded? | has replies from others | from original author | other observations                                          |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- | --------------------- | --------------------------- | ----------------------------------------- | --------- | ----------------------- | -------------------- | ----------------------------------------------------------- |
| New Paper              | https://twitter.com/EJohnWherry/status/1182004777510531074   | Zeyu Chen's new paper in Immunity identifies early precursors of Exhausted T cells and identifies underlying molecular circuitry. Helps clarify the developmental trajectories of Tex and relationship of Tex Precursors, Progenitor and terminal Tex subsets. | https://t.co/06V8UvfjCP?amp=1 | 2                     | 1                           | 1                                         | 0         | 2                       | 0                    | some cool discussion on the methods in the comments         |
| New Paper              | https://twitter.com/jpjhall/status/1184437346399408128       | New paper: plasmid costs can be ameliorated during the process of transconjugant colony growth. | None                          | 2                     | 0                           | 1                                         | 1         | 0                       | 1                    | also discusses implications + meta discussion about methods |
| New Paper              | https://twitter.com/BrockhurstLab/status/1184423614281637888 | How fast?! Very very fast! Compensatory evolution to ameliorate plasmid fitness cost within hours (with movies) by 3 diverse mechanisms Exciting work led by ⁦ | None                          | 0                     | 0                           | 1                                         | 0         | 1                       | 0                    | tags the original authors                                   |

## **Our findings by now**

1. Most Tweets don’t have many replies, among those received replies, some are congrats;

2. Among the tweets have graphs, some of the graphs are the screenshots of the paper, which usually are titles/authors/abstracts.

3. Some of the tweets are from the official account of the labs/Institutions/NGOs.

4. Some of the tweets only mentioned the paper being published and nothing about the contents of the paper.

5. Most tweets mentioned at least one of the problems/solutions about the paper, a few did the both.

6. For those key words such as Our research and Our finding usually contain the users’ own work.
7. Comments for those Tweets are usually none or Congrats.

## **Next Step**

Building a data pipeline through Twitter API, collecting tweets for mass data analysis.

Based on the large data collected, made further examinations and conclusions, finished an academic paper.

## **My Role**

As one of two RAs in Professor Joel Chan’s [lab](http://joelchan.me/lab/) working on this project. I am responsible for collecting data on Twitter and building database, and summarized findings based on the data collected.

