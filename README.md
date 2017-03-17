# Example submission for the hackathon

This is an example submission in Python that you can use as reference/starting point to create your own submission.

## Content

* **Dockerfile**

This is where you specify the container environment, define the base image, install dependencies for your code and run the script that executes the submission. Simply modify this for your needs. Here is how a different Dockerfile for Python could look like, which installs all common data science libraries:

```
# defines what base image to use
FROM python:3

# install some dependencies needed for scipy
RUN apt-get update
RUN apt-get install -y liblapack-dev libblas-dev gfortran

# install your solution’s dependencies (you can add more here)
RUN pip3 install NumPy
RUN pip3 install SciPy
RUN pip3 install sklearn
RUN pip3 install pandas

# cleanup to reduce container size
RUN apt-get purge -y gfortran
RUN apt-get -y autoremove

# add everything in the current local folder to the container: REQUIRED!
ADD . /

# run your submission by executing the entrypoint: REQUIRED!
ENTRYPOINT ["./run.sh"]
```

Make sure `ADD . /` and `ENTRYPOINT ["./run.sh"]` are included in your Dockerfile.

* **data/**

This is the folder in which the data will be provided when running your code in the cloud. Therefore, your code should read the data files from this folder. We included sample data files that can be used to locally run your code and test that everything is run corretly.

* **example_result.txt**

This is an example result file to demonstrate the required format of the result predictions. Your code needs to write its predictions in the same format in a file called `result.txt`.

* **train_predict.py**

This is the actual submission code and the part you need to replace. This needs to be an executable that does everything you want your code to do in order to build a model and produce predictions. This code essentially needs to read in the data, make predictions and write those into `result.txt` in the above specified format. In this particular example it reads in the `tracking.csv` data (remember: from the folder `data/`), predicts symptoms based on past averages, and writes the predictions in the above specified format to a file called `result.txt` (important!).

* **run.sh**

This is the entrypoint of the Docker container and the one the executes your code (here: `train_predict.py`). Usually just a one or two liner. You can run `run.sh` locally before submitting your solution in order to check that everything runs as expected. If your submission can run locally and generate a file `result.txt` then it's probably working fine.

* **submit.py**

Run `./submit.py` to run some basic integrity tests and submit your solution to Statice.

    $./submit.py
      competition username:test@helloclue.com
      competition password:<my_competition_password>
      Successfully submitted.
      Check http://statice.wattx.io/submissions for rating.
    $

* **test/**

Don't bother about this folder, it just contains a test script that is being used by `submit.py`. Simply leave it as part of your submission.
