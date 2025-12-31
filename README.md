# ANASCAT

## Overview
Recup le script de Theo & run on anapath image
  - cluster tile composition
  - tile classification
  - use for segmentation

## Installation
Need to install the special lib from sheng & menard
'''
git clone https://github.com/SihaoCheng/scattering_transform/ 
cp -r scattering_transform/scattering .
'''

## Workflow - Basic
  - [x] Load img
  - [x] extract patches
  - [x] run scatering
  - [x] normalize
  - [x] umap
  - [x] kmean
  - [x] plot umap
  - [x] display cluster on orgin image

## TODO SEG
  - [x] craft seg1 img
  - [x] run patch scat clustering
  - [x] plot cluster in figure
  - [x] identify border cluster
  - [x] craft segmentation mask
  - [x] refactor code for segmentation purpose
  - [x] create new sample image
  - [x] test
  - [ ] rerun kmeans option
  


