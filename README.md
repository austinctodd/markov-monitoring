# markov-monitoring
Simulate data coming from a hypothetical sensor that is able to detect react to the sensor values being above a certain level.  The model will run in the background and push data to a database.  Then, the dashboard will read the database information as a streaming job, plotting the data and notifying the user of the current "level" (i.e., active functioning, warning, danger). 
