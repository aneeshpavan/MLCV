# Submission Instructions

Students should submit their assignments only using the files provided. While most problems need students to write the library functions in Python script, the final problem on MNIST classifier performance evaluation should be written in the Jupyter notebook provided, which is labeled as "HW2_MNIST_NN4.ipynb". Students are not allowed to use any deep learning libraries (e.g. PyTorch, Tensorflow, Keras, CAFFE, Jax and other similar packages). The problem statement can be found in this [link](https://sid-nadendla.github.io/teaching/SP2024_MLCV/HWs/HW2_MLCV_SP2024.pdf).

NOTE: You may reuse your code from your HW1 submission as needed.

# Some Useful References

## Enabling GPUs for Data Parallelism
For your reference, thee Jupyter Notebook used by the GTA in the recitation lecture on Decorators, CUDA and Jit is also included in this repository. This file will not be considered an integral part of your HW2 submission. 

## To understand GPU architecture and using numba package for CUDA users
* https://numba.readthedocs.io/en/stable/cuda/index.html
* https://colab.research.google.com/github/cbernet/maldives/blob/master/numba/numba_cuda.ipynb
* https://thedatafrog.com/en/articles/boost-python-gpu/

# Instructions for Cloning hw2 folder

Students will be given access to the Git repository as 'developers'. As a result, they can clone the master branch and submit their respective assignments by following the procedure given below:

## Execute once, to clone a repository:
```
$ git clone https://git-classes.mst.edu/2022-SP-CS6406/<repository_name>.git
```

## Execute as many times as you like from within the directory/repository you cloned to your hard drive (just an example):
```
# To check the status of your repository:
$ git status

# To stage/add a file:
$ git add *.py *.pdf *.md

# To add a folder:
$ git add SUBDIRECTORY/*

# To commit changes and document them:
$ git commit -m "Informative description of the commit"

# To submit your assignments:
$ git push
```


## Do not add:
Compiled or generated files like *.out, *.log, *.syntex.gz, *.bib, your executable files, etc. Put the name of these files in a text file named .gitignore

If you see your changes reflected on the git-classes site, you have submitted successfully.

## Useful links:
[Git Cheatsheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)

[Videos on Git basics](https://git-scm.com/videos)
