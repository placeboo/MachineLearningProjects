
# To Start
1. Clone the repository
2. Download the datasets from the links provided below in [Dataset](#Dataset) section
3. Create the environment ```conda env create -f environment.yml```
4. Set the working directory to the root of the repository in the terminal, which is the directory containing the `main_dateset1.py` and `main_dateset2.py`.
4. Run the code. The input parameters are saved in the `dataset1_config` and `dataset_config` folders. You can change the parameters in the file to run the code with different parameters.
To run the model, use the following command:
```python
python main_dataset1.py -c {config file path} -m {model_name}
```
The model name can be one of the following:
- nn: Neural Network
- svm: Support Vector Machine
- knn: K-Nearest Neighbors
- boosting: Gradient Boosting Trees
One example of the command is:
```python
python main_dataset2.py -c dataset2_config/config.yaml -m nn
```

# Dataset

## Bank Marketing Campaign Subscription
Source: [Download from Kaggle](https://www.kaggle.com/datasets/pankajbhowmik/bank-marketing-campaign-subscriptions)

### Description
The dataset contains information about marketing campaigns that were conducted via phone calls from a Portuguese banking institution to their clients. Purpose of these campaigns is to prompt their clients to subscribe for a specific financial product of the bank (term deposit). After each call was conducted, the client had to inform the institution about their intention of either subscribing to the product (indicating a successful campaign) or not (unsucessful campaign).
The final output of this survey will be a binary result indicating if the client subscribed ('yes') to the product or not ('no').

The dataset has 41188 rows (instances of calls to clients) and 21 columns (variables) which are describing certain aspects of the call. Please note that there are cases where the same client was contacted multiple times - something that practically doesn't affect the analysis as each call will be considered independent from another even if the client is the same.

### Data Schema
The predictor variables (features) contained in the dataset can be divided into the following five sections:

1. Variables that describing attributes related directly to the client:
    - age
    - job: type of job (e.g. 'admin', 'technician', 'unemployed', etc)
    - marital: marital status ('married', 'single', 'divorced', 'unknown')
    - education: level of education ('basic.4y', 'high.school', 'basic.6y', 'basic.9y','professional.course', 'unknown','university.degree','illiterate')
    - default: if the client has credit in default ('no', 'unknown', 'yes')
    - housing: if the client has housing a loan ('no', 'unknown', 'yes')
    - loan: if the client has a personal loan ? ('no', 'unknown', 'yes')
2. Variables related to the last contact of the current campaign:
    - contact: type of communication ('telephone', 'cellular')
    - month: month of last contact
    - day_of_week: day of last contact
    - duration: call duration (in seconds)

3. Other variables related to the campaign(s):
    - campaign: number of contacts performed during this campaign and for this client 
    - pdays: number of days passed by after the client was last contacted from a previous campaign 
    - previous: number of contacts performed before this campaign and for this client 
    - poutcome: outcome of previous marketing campaign ('nonexistent', 'failure', 'success')

4. Socioeconomic variables:
    - emp.var.rate: employement variation rate - quarterly indicator
    - cons.price.idx: consumer price index - monthly indicator
    - cons.conf.idx: consumer confidence index - monthly indicator
    - euribor3m: euribor 3 month rate - daily indicator 
    - nr.employed: number of employees - quarterly indicator

5. Output variable (desired target): subscribed, 'Yes' or 'No'
## YouTube Comments Spam
- [Youtube Comments Spam Dataset](https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset)
