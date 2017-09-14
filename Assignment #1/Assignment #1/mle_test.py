from q1_create_n_gram import create_n_gram, create_n_gram_wcount
import math as m

sum_data = open('sum_train.txt', 'r', encoding='utf8').read().split('*****')
edu_data = open("edu_train.txt", "r", encoding='utf8').read().split('*****')
exp_data = open("exp_train.txt", "r", encoding='utf8').read().split('*****')

size = len(exp_data)
split_coeff = 0.7
train_size = round(size*split_coeff)

sum_train, edu_train, exp_train = sum_data[:train_size], edu_data[:train_size], exp_data[:train_size]
sum_test, edu_test, exp_test = sum_data[train_size:], edu_data[train_size:], exp_data[train_size:]

# train stage
# it was decided to use 2-grams

two_gram_all, two_gram_all_count = create_n_gram_wcount(sum_train + edu_train + exp_train, 2)
two_gram_sum, two_gram_sum_count = create_n_gram_wcount(sum_train, 2)
two_gram_edu, two_gram_edu_count = create_n_gram_wcount(edu_train, 2)
two_gram_exp, two_gram_exp_count = create_n_gram_wcount(exp_train, 2)

#test stage

two_gram_list = dict(sum=two_gram_sum, edu=two_gram_edu, exp=two_gram_exp)

def classify(segment):
    prob_label = {}
    for two_grams in two_gram_list:
        log_prob = 0
        for n_gram in segment:
            if n_gram in two_gram_list[two_grams]:
                log_prob += m.log(two_gram_list[two_grams][n_gram]/(3*two_gram_all[n_gram]))
            elif n_gram in two_gram_all:
                log_prob -= m.log(two_gram_all[n_gram])
            else:
                log_prob -= m.log(1/two_gram_all_count)
        prob_label[two_grams] = log_prob
    return min(prob_label, key = lambda classLabel: prob_label[classLabel])

def test(test_data, true_value):
    accuracy = 0
    for t in test_data:
        accuracy += 1/len(test_data) if classify(create_n_gram([t], 2).keys())==true_value else 0
    return accuracy

print('Summary accuracy: ' + str(round(test(sum_test, 'sum'), 3)))
print('Education accuracy: ' + str(round(test(edu_test, 'edu'), 3)))
print('Experience accuracy: ' + str(round(test(exp_test, 'exp'), 3)))