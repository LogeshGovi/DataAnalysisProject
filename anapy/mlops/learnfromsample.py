from ..sampling.RandomSampling import RandomSampling as rss
from ..sampling.SystematicSampling import SystematicSampling as sss
from ..sampling.StratifiedSampling import StratifiedSampling as srss
from ..sampling.ClusterSampling import ClusterSampling as css
from sklearn.preprocessing import StandardScaler
import time




class LearnFromSample:
    def learn_from_sample(train_data,train_target,test_data, test_target,
                           sample_size, SamplingMethod,classifier,
                           scaler=None, replacement=None, strat_col = None):
        if SamplingMethod == 'random':
            sample_train_data, sample_train_target = rss.get_random_samples_np(train_data, train_target,sample_size,replacement)
        if SamplingMethod == 'systematic':
            sample_train_data, sample_train_target = sss.systematic_sample(train_data,train_target,sample_size)
        if SamplingMethod == 'stratified':
            sample_train_data, sample_train_target = srss.stratified_sample(train_data,train_target,strat_col,sample_size,replacement)
        if SamplingMethod == 'cluster':
            sample_train_data, sample_train_target = css.cluster_sample(train_data,train_target,sample_size,replacement)
        te_data = test_data
        if  isinstance(scaler,StandardScaler) == True and scaler!=None:
            sample_train_data = scaler.transform(sample_train_data)
            te_data = scaler.transform(te_data)
        clf = classifier
        # classifier fit time
        t_start_classifier = time.time()
        clf.fit(sample_train_data,sample_train_target)
        t_end_classifier = time.time()
        classifier_fit_time = t_end_classifier - t_start_classifier

        train_pred = clf.predict(sample_train_data)
        # classifier predict time
        t_start_prediction = time.time()
        test_pred = clf.predict(te_data)
        t_end_prediction = time.time()
        classifier_prediction_time = t_end_prediction - t_start_prediction
        train_true = sample_train_target
        test_true = test_target
        return [train_true,train_pred], [test_true, test_pred],[classifier_fit_time,classifier_prediction_time]





