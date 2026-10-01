# Dual Vision Analyzer

Bachelor-thesis web app: a convolutional neural network benchmarked against a classical HOG + SVM pipeline for image classification, with both approaches explained side by side and a live classifier running in the browser.

**Live:** https://dual-vision-analyzer.vercel.app  
**Thesis:** https://doi.org/10.5281/zenodo.22788781

## Result

CIFAR-10, seed 42, reproducible run:

| Model | Accuracy |
|---|---|
| CNN | **70.02 %** |
| HOG + SVM | 51.66 % |

The CNN learns its features; HOG hand-crafts them. The gap is the thesis.

## What the app does

- Upload any image and classify it in the browser with MobileNetV2 (TensorFlow.js) — no server, no upload
- Side-by-side explanation of the two pipelines: convolution + pooling vs gradient histograms + linear SVM
- Accuracy-gap chart and discussion of where each approach wins
- Dark / light theme; defensive handling of bad images and model-load failures

## Stack

Next.js · React · TensorFlow.js · MobileNetV2 · Tailwind CSS · deployed on Vercel

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.
