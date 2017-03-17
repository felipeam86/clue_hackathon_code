## Fork Felix' repo
On Github, go to felix repo: https://github.com/aktivkohle/clue_hackathon_code
Fork felix repo so you have a copy on your github (fork icon top right corner of the screen)

## Clone your own version of felix' repo
On your machine, create a local instance of the forked repo (from your github)
e.g. git clone git@github.com:GregVial/clue_hackathon_code.git

## Setup link between your local version and Felix (upstream)
On your machine, create a link with Felix repo
e.g. git remote add upstream git@github.com:aktivkohle/clue_hackathon_code.git

You are all set

## Push: always to your github
git add ...
git commit -m "description of change"
git push origin master

The go to your github and make a pull request (merge) that Felix will accept of reject

## Pull: always from felix repo
Whenever new changes are pushed to the repo (from other team):
git pull upstream master
