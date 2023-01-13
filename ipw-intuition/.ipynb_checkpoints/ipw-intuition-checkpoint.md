```python
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.max_colwidth', None)
```


```python
import warnings
warnings.filterwarnings("ignore")
from plotnine import *
from tqdm import tqdm
from joblib import Parallel, delayed # for parallel processing
tqdm._instances.clear() 
```

## Create Campaign Data

Both stores have same number of members. We can get an unbiased estimate of the population mean spending among customers if the number of treated is the same in both stores.


```python
true_mu1treated = 20
true_mu2treated = 40
true_mutreated = true_mu1treated*0.5 + true_mu2treated*0.5
true_mutreated
```




    30.0




```python
true_mu1control = 10
true_mu2control = 30
true_mucontrol = true_mu1control*0.5 + true_mu2control*0.5
true_mucontrol
```




    20.0




```python
true_tau = true_mutreated - true_mucontrol
true_tau
```




    10.0




```python
np.random.seed(88)

def run_campaign():
    n, p , obs = 1, .5 , 2000 # number of trials, probability of each trial, number of observations
    store = np.random.binomial(n, p, obs)+1
    df = pd.DataFrame({'store':store})

    probtreat1 = .5
    probtreat2 = .9
    
    treat = lambda x: int(np.random.binomial(1, probtreat1, 1)) if x==1 else int(np.random.binomial(1, probtreat2, 1)) 
    
    spend = lambda x: float(np.random.normal(true_mu1treated, 3, 1)) if (x[0]==1 and x[1]==1) else ( float(np.random.normal(true_mu2treated, 3, 1) ) if  (x[0]==2 and x[1]==1)   
                                                                                              else (float(np.random.normal(true_mu1control, 2, 1) ) if  (x[0]==1 and x[1]==0)  
                                                                                              else  float(np.random.normal(true_mu2control, 2, 1))     ))

    df['treated'] = df['store'].apply(treat)
    df['spend'] = df[['store','treated']].apply(tuple,1).apply(spend)
    
    prob1 = df.query('store==1').shape[0]/df.shape[0]
    prob2 = df.query('store==2').shape[0]/df.shape[0]
    
    wrong_value_treated = np.mean(df.query('treated==1')['spend'])
    wrong_value_control = np.mean(df.query('treated==0')['spend'])
    
    wrong_tau = wrong_value_treated - wrong_value_control
    
    est_mu1treated = np.mean(df.query('treated==1 & store==1')['spend'])
    est_mu2treated = np.mean(df.query('treated==1 & store==2')['spend'])

    right_value_treated = prob1*est_mu1treated + prob2*est_mu2treated
        
    est_mu1control = np.mean(df.query('treated==0 & store==1')['spend'])
    est_mu2control = np.mean(df.query('treated==0 & store==2')['spend'])
    
    right_value_control = prob1*est_mu1control + prob2*est_mu2control
    
    right_tau = right_value_treated - right_value_control
    
    #estimate propensity score:
    ps1 = df.query('treated==1 & store==1').shape[0]/df.query('store==1').shape[0]
    ps2 = df.query('treated==1 & store==2').shape[0]/df.query('store==2').shape[0]

    df['ps'] = pd.Series(np.where(df['store']==1, ps1, ps2))

    ipw_value_treated = np.mean( (df['spend']*df['treated'])/df['ps'])
    ipw_value_control = np.mean( (df['spend']*(1-df['treated']) )/(1-df['ps'] ))
    ipw_tau = ipw_value_treated - ipw_value_control
    
    return [wrong_value_treated, right_value_treated, ipw_value_treated, wrong_value_control, right_value_control, ipw_value_control , wrong_tau, right_tau, ipw_tau]
```


```python
run_campaign()
```




    [33.18465520598871,
     30.21741176723726,
     30.21741176723726,
     13.050984691822144,
     20.053585790881716,
     20.053585790881723,
     20.133670514166568,
     10.163825976355543,
     10.163825976355536]



## Run Simulation


```python
tqdm._instances.clear() 
sim = 1000

values = Parallel(n_jobs=4)(delayed(run_campaign)() for _ in tqdm(range(sim)) )
results_df = pd.DataFrame(values, columns=['wrong_treat','right_treat','ipw_treat','wrong_control','right_control','ipw_control','wrong_tau','right_tau','ipw_tau'])

for c in results_df.columns.tolist():
    print(f"Mean value computed {c}: {round(results_df[c].mean(),3)}, 95% C.I.: {(round(np.percentile(results_df[c], 2.5),3), round(np.percentile(results_df[c], 97.5),3))}")
```

    100%|██████████████████████████████████████| 1000/1000 [00:03<00:00, 260.88it/s]


    Mean value computed wrong_treat: 32.857, 95% C.I.: (32.297, 33.381)
    Mean value computed right_treat: 29.992, 95% C.I.: (29.502, 30.486)
    Mean value computed ipw_treat: 29.992, 95% C.I.: (29.502, 30.486)
    Mean value computed wrong_control: 13.315, 95% C.I.: (12.77, 13.96)
    Mean value computed right_control: 19.987, 95% C.I.: (19.494, 20.517)
    Mean value computed ipw_control: 19.987, 95% C.I.: (19.494, 20.517)
    Mean value computed wrong_tau: 19.542, 95% C.I.: (18.763, 20.329)
    Mean value computed right_tau: 10.005, 95% C.I.: (9.731, 10.274)
    Mean value computed ipw_tau: 10.005, 95% C.I.: (9.731, 10.274)


## Plot Results


```python
results_df_treat_long = pd.melt(results_df, value_vars=['wrong_treat','ipw_treat'])
```


```python
p_treat = (ggplot(results_df_treat_long, aes(x='value' ,fill='variable'))+
 geom_histogram(alpha=.75, bins=50) +
  xlab('mean spending') + ylab('Count') +
 labs(title="Campaign Results for Treated",
      caption = "Note: Black dashed lines represents the true mean                                                                      ")+
 # geom_vline(xintercept=ates.mean(), 
 #           colour='black', linetype='dashed' ) + 
 geom_vline(xintercept=true_mutreated, colour='black', linetype='dashed' ) + 
  theme(figure_size=(10, 10)) +
  theme(axis_text_x = element_text(angle = 45, hjust = 1))
)
p_treat
```


    
![png](output_13_0.png)
    





    <ggplot: (708673052)>




```python
results_df_control_long = pd.melt(results_df, value_vars=['wrong_control','ipw_control'])
```


```python
p_control = (ggplot(results_df_control_long, aes(x='value' ,fill='variable'))+
 geom_histogram(alpha=0.75, bins=50) +
  xlab('mean spending') + ylab('Count') +
 labs(title="Campaign Results for Control",
      caption = "Note: Black dashed lines represents the true mean                                                                      ")+
 # geom_vline(xintercept=ates.mean(), 
 #           colour='black', linetype='dashed' ) + 
 geom_vline(xintercept=true_mucontrol, colour='black', linetype='dashed' ) + 
  theme(figure_size=(10, 10)) +
  theme(axis_text_x = element_text(angle = 45, hjust = 1))
)
p_control
```


    
![png](output_15_0.png)
    





    <ggplot: (708830221)>




```python
results_df_tau_long = pd.melt(results_df, value_vars=['wrong_tau','ipw_tau'])
```


```python
p_tau = (ggplot(results_df_tau_long, aes(x='value' ,fill='variable'))+
 geom_histogram(alpha=0.75, bins=50) +
  xlab('mean spending') + ylab('Count') +
 labs(title="Campaign Results",
      caption = "Note: Black dashed lines represents the true mean                                                                      ")+
 # geom_vline(xintercept=ates.mean(), 
 #           colour='black', linetype='dashed' ) + 
 geom_vline(xintercept=true_tau, colour='black', linetype='dashed' ) + 
  theme(figure_size=(10, 10)) +
  theme(axis_text_x = element_text(angle = 45, hjust = 1))
)
p_tau
```


    
![png](output_17_0.png)
    





    <ggplot: (708854541)>


