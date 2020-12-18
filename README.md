# Adopting Learning-to-rank Algorithm forReviewer Recommendation #

**Requirements**
* Python 3.5 or newer
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib%20Installation/) library 2.14 or newer

**Dataset**

The full dataset is hosted on [Onedrive](https://queensuca-my.sharepoint.com/:f:/g/personal/17gz2_queensu_ca/EhUGPglYfAFPs-R9p69va_ABsbA3uGYu743AStjXqx56GQ?e=JMx3FT)

*  **Reviewer_PullRequests** contains the CSV format data for 80 studied GitHub projects. Each folder in Reviewer_PullRequests contains the expertise features of reviewers regarding pull requests for each GitHub project.  

**Files description**
*  **data** folder contains the sample data of the reviewer recommendation data that are already present in the Onedrive link.
    * **sample.csv** contains the sample of the reviewers' expertise features data.
*  **python scripts** folder contains the scripts for using Learning-to-rank (LtR) models to learn weights of expertise features and evaluating the performance of the LtR models in recommending reviewers.
    * **reviewer_recommendation.py** contains the script for building and evaluating LtR models.
*  **Appendix.pdf** contains the appendix of the paper.

For any questions, please send email to g.zhao@queensu.ca
