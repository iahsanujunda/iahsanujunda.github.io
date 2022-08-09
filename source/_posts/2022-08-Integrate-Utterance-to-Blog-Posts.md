---
title: Integrate Utterance to Blog Posts
date: 2022-08-08 21:29:58
tags:
---

[Utterance](https://utteranc.es/) is a comment widget that leverage github issues to bring discussion to our blog. I have been looking for a discussion tool for a while now, but none seems suitable. Lots of hexo-powered blogs use disqus for discussion, however I felt overwhelmed with the many configuration of disqus. The appearance is very cluttered as well. I just need a simple discussion widget that will let user identify themselves before commenting, but hexo as a static page generator doesn't make use any kind of database storage so authenticating would need to be taken care of by third party.

Come utterance, brought to my attention when using this [cactus theme](https://github.com/probberechts/hexo-theme-cactus). Once I read the documentation, I am immediately sold. Utterance make use of github Oauth to authenticate users, it store comments as github repo issues, and to integrate utterance to our site, we just need to add the following script to our html.

```html
<script src="https://utteranc.es/client.js"
        repo="iahsanujunda/iahsanujunda.github.io"
        issue-term="pathname"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```

Or in the case of cactus theme, modify the following configuration

```yaml
utterances:
  enabled: true
  repo: iahsanujunda/iahsanujunda.github.io
  issue_term: pathname
  label: Comment
  theme: github-dark
```

How delightful!

And to top it all off, the appearance is sleek and simple. What's not to love.
