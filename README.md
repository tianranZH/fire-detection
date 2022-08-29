# fire-detection
Is there any fires in the last 15mins?
 an end-to-end project to illustrate fire detection algorithm


1. get pairs VIIRS I4/I5 h5 data from aws (https://registry.opendata.aws/noaa-jpss/)
2. run through a basic fire algorithm.
3. report total fires, and a plot of locations on a world map.
4. save the data in a database in RDS
5. deploy fire algorithm in lambda .


# Deploy Lambda functions
For lambda function deployment, I am following the AWS guidance here (https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/deploy-lambda-functions-with-container-images.html)
1. I use github to manage code development.
2. AWS CodeBuild is used to auto-build image when there is any push happening on main branch.
3. CodeBuild publishes the image to Amazon ECR.
4. Lambda function will run use the image in ECR.