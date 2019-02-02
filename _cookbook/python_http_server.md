---
layout: post
title:  "Run python http server in any directory"
date:   2019-02-02 11:53:35 +0100
categories: python http server
---
If you are working on simple html javascript page and need to set up http server quickly, python has great one-line solution.

Open terminal in directory where you want to setup http server and type:
```
python3 -m http.server 8081
```

This will start http server on port 8081.
This is useful when you need to:
* test ajax call
* serve static files
    