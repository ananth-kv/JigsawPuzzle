# JigsawPuzzle

Jigsaw Puzzle solver on Imagenet dataset

Based on the paper: Unsupervised Learning of Visual Representions by solving Jigsaw Puzzles

The data preprocessing is carried out exactly how it is done in the cited paper.

Siamese Network with 9 branches. Each branch is an Alexnet w/o fully connected layer.


* The original work is here: [JigsawPuzzleSolver](http://www.cvg.unibe.ch/research/JigsawPuzzleSolver.html)
* Code Skeleton was taken from here: [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)
