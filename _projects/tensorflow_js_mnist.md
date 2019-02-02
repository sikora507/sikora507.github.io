---
layout: post
title:  "TensorflowJS mnist real-time classification"
date:   2019-02-02 11:53:35 +0100
categories: tensorflowjs mnist image recognition keras python
---
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.14.2/dist/tf.min.js"></script>
<h1>Tensorflow JS - recognizing drawn digits</h1>
<p>Draw digits on the canvas and test tensorflow.js predictions in realtime.</p>
<div class="row">
    <div class="col-sm-6"><canvas id="canv" width="28" height="28" style="width:280px; height:280px; border:1px solid black;"></canvas></div>
    <div class="col-sm-6" style="font-size: 72px">Predicted: <span id="prediction"></span></div>
</div>
<br />
<button class="btn" id="clearBtn">Clear canvas</button>
<p>Model was trained in tensorflow.keras using MNIST digits data set.
    Then it was exported to tensorflow.js format to be able to use it
    the browser in realtime to make predictions.
</p>
<script type="text/javascript">
    function makeArray(w, h, val) {
        var arr = [];
        for (i = 0; i < h; i++) {
            arr[i] = [];
            for (j = 0; j < w; j++) {
                arr[i][j] = val;
            }
        }
        return arr;
    }
    function getPoint(e, rect) {
        return {
            x: (e.clientX - rect.left) / 10,
            y: (e.clientY - rect.top) / 10,
            width: 1
        };
    }

    (async function l() {
        const MODEL_URL = '/assets/models/tfjs_mnist_1/model.json';
        const model = await tf.loadModel(MODEL_URL);
        console.log(model);
        console.log("model loaded");

        var pixelArray = makeArray(28, 28, 0);
        var el = document.getElementById('canv');
        var btn = document.getElementById('clearBtn');
        var predictionContainer = document.getElementById('prediction');
        var ctx = el.getContext('2d');
        var rect = el.getBoundingClientRect();
        ctx.lineJoin = ctx.lineCap = 'round';

        var isDrawing;
        el.onmousedown = function (e) {
            rect = el.getBoundingClientRect();
            isDrawing = true;
            var point = getPoint(e, rect);
            ctx.moveTo(point.x, point.y);
            ctx.beginPath();
        };

        el.onmousemove = function (e) {
            if (!isDrawing) return;
            var point = getPoint(e, rect);
            ctx.lineTo(point.x, point.y);
            ctx.stroke();
        };

        el.onmouseup = function () {
            isDrawing = false;
        };

        btn.onmouseup = function () {
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        }

        setInterval(function () {
            (async function () {
                if (isDrawing) {
                    imageData = ctx.getImageData(0, 0, 28, 28);
                    for (let y = 0; y < 28; ++y) {
                        for (let x = 0; x < 28; ++x) {
                            pixelArray[y][x] = imageData.data[(28 * 4 * y) + (x * 4 + 3)] / 255;
                        }
                    }
                    let predictions = tf.tidy(() => {
                        let tensor = tf.tensor(pixelArray).reshape([1, 28, 28, 1]);
                        return model.predict(tensor).argMax(1);
                    });
                    predictionContainer.innerText = predictions.dataSync()[0];
                }
            })();
        }, 500);
    })();
</script>

Github URL:
<https://github.com/sikora507/python-keras-tensorflowjs-mnist>
<br/>
<br/>
<br/>
<br/>
<br/>
Also if you haven't already, I strongly recommend checking out this excelent book by Aurélien Géron. It's still up-to-date and might be usefull even if you are already familiar with machine learning.

<a target="_blank"  href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=2afda09226851f789f24d5671b6d1ca3"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1491962291&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=sikora507-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

<a target="_blank" href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=64bd5103ef0b10b13cba3936d277e786">Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />