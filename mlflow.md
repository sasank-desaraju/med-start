# MLFlow for Experiment Tracking
## Why and how we are doing this



During training, log to a specified MLFlow directory. This does not require running a server or anything.
Whenever you want to know the results, you can start the MLFlow server and see the results in the browser.
This is done by starting the server in a HPG terminal session and pointing the server to the folder where you've been saving all your stuff.
This folder should be in /blue/group_name/username/mlflow I think. Then, it'll populate that with the experiment names as it sees fit.

So to see the results, start a server by executing the bash file mlflow_server.sh, which you might need to change to fit your needs.
You can then see the results in the browser by going to the server's address and port. This is usually something like http://localhost:5000 .

So, doing this while having a VSCode Tunnel to HPG is making it so VSCode asks if you want to open the connection, which it automatically finds, in the browser. This is very convenient.
I'm currently doing this with tmux to use two terminal sessions, one for vscode_tunnel and the other for the mlflow server. This is suboptimal to require of people who can't use tmux, so I'm looking into how to make this easier.
I guess you could run the mlflow_server.sh script through a terminal session within VSCode.
Oh snap that worked!!!

Okay, so for non-power users, **just do the normal workflow of getting a VSCode Tunnel to HPG and then run the mlflow_server.sh script in the terminal session.**
Great.