FROM public.ecr.aws/lambda/python:3.10


COPY requirements.txt .

# install dependencies
RUN pip install --user -r requirements.txt

# copy function code
COPY src/* ${LAMBDA_TASK_ROOT}

# set cmd to handler
CMD ['app.lambda_handler']
