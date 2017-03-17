## Fork Felix' repo
On Github, go to felix repo: https://github.com/aktivkohle/clue_hackathon_code
Fork felix repo so you have a copy on your github (fork icon top right corner of the screen)

## Clone your own version of felix' repo
On your machine, create a local instance of the forked repo (from your github)
e.g. 

_git clone git@github.com:GregVial/clue_hackathon_code.git_

## Setup link between your local version and Felix (upstream)
On your machine, create a link with Felix repo
e.g. 
_git remote add upstream git@github.com:aktivkohle/clue_hackathon_code.git_

You are all set

## Push: always to your github
_ git add ..._ 

_git commit -m "description of change"_

_git push origin master_

The go to your github and make a pull request (merge) that Felix will accept of reject

## Pull: always from felix repo
Whenever new changes are pushed to the repo (from other team):

_git pull upstream master_
