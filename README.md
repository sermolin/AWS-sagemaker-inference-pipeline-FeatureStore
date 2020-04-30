# Setup instructions

 1. Run the cloudformation script. You'll need to create a Secret in AWS to store your github access credentials. See here for futher detail: https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-git-repo.html
 2. Log into the AWS console, open Cloudformation. Find the stack that was just created, and open the resources tab to find the name of the dynamodb table you just created
 3. Open DynamoDB from the AWS console and find the table. Add a item with the following data:
    {
      "ID": "1",
      "shell_weight": "0.15",
      "shucked_weight": "0.2245",
      "viscera_weight": "0.101",
      "whole_weight": "0.514"
    }
 4. Open Jupyter on the newly created instance. Open the `sklearn_abalone_featurizer.py` file and find the line `feature_store_table = 'feature_store'`. Update this variable with the table name you looked up in step 2.
 5. Launch the Jupypter notebook. Click on: `restart and run all cells`



