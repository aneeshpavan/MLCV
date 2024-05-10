# Submission Instructions

Students should submit their assignments only using the files provided, which are labeled as "HW3_Question#.ipynb". In addition, the jupyter notebook titled "R6_Transfer_Learning.ipynb" used by Mukund Telukunta (GTA) in his recitation lecture is also attached as a reference for students.

# Some Useful References

## Enabling GPUs for Data Parallelism
For your reference, thee Jupyter Notebook used by the GTA in the recitation lecture on Decorators, CUDA and Jit is also included in this repository. This file will not be considered an integral part of your HW2 submission. 

## PyTorch documentation and other resources
* PyTorch: https://pytorch.org/docs/stable/index.html
* Pretrained Models in PyTorch: https://pytorch.org/vision/master/models.html
* Transfer Learning (Example from a similar course at Stanford): https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 
* Foundry: https://wiki.itrss.mst.edu/dokuwiki/pub/foundry 
* Pascal VOC Dataset: http://host.robots.ox.ac.uk/pascal/VOC/ 

# Instructions for Cloning hw3 folder

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
