# fire-detection
Is there any fires in the last 15mins?
 an end-to-end project to illustrate fire detection algorithm


1. get pairs VIIRS I4/I5 h5 data from aws (https://registry.opendata.aws/noaa-jpss/)
2. run through a basic fire algorithm
3. report total fires, and a plot of locations on a world map
4. save the data in a database in RDS
5. deploy fire algorithm in lambda 