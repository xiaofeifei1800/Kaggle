import csv
import cPickle
import feature_fun

def processDataMK(in_file_name, cust_dict, lag_cust_dict,total_dict,lag1,lag2,lag3,lag12,lag_feature):
    x_vars_list = []
    y_vars_list = []

    for row in csv.DictReader(in_file_name):
        if row['fecha_dato'] not in ['2015-01-28','2015-02-28','2015-03-28', '2015-04-28','2015-05-28', '2015-06-28',
                                      '2015-12-28'
                                     '2016-01-28','2016-02-28','2016-03-28', '2016-04-28','2016-05-28', '2016-06-28']:
            continue
        #Leave out first month
        cust_id = int(row['ncodpers'])
        #print(row['fecha_dato'])
        if (row['fecha_dato'] in ['2015-01-28', '2016-01-28'] ):
            target_list = feature_fun.getTarget(row)
            lag_cust_dict[cust_id] =  target_list[:]
            lag_feature[cust_id] = [feature_fun.getAge(row)]
            sex = feature_fun.getIndex(row, 'sexo')
            age = feature_fun.getAge(row)
            income = feature_fun.getRent(row)
            lag_feature[cust_id].append(income)
            lag_feature[cust_id].append(feature_fun.getMonth(row))
            lag_feature[cust_id].append(feature_fun.getCustSeniority(row))
            lag_feature[cust_id].append(feature_fun.getMarriageIndex(row, age, sex, income) )


            lag_feature[cust_id].append(feature_fun.sex_activity(row))
            lag_feature[cust_id].append(feature_fun.getTotalTarget(row))
            lag_feature[cust_id].append(feature_fun.getCustSeniority_group(row))

            lag_feature[cust_id].append(feature_fun.getjoinMonth(row))
            lag_feature[cust_id].append(feature_fun.income_cate(row))
            lag_feature[cust_id].append(feature_fun.getAge_group(row))
            lag_feature[cust_id].append(feature_fun.new_cust(row))
            continue

        if (row['fecha_dato'] in ['2015-12-28'] ):
            target_list = feature_fun.getTarget(row)
            lag12[cust_id]= target_list[:]
            continue

        if (row['fecha_dato'] in ['2015-04-28', '2016-04-28'] ):
            target_list = feature_fun.getTarget(row)
            lag1[cust_id]= target_list[:]
            continue

        if (row['fecha_dato'] in ['2015-03-28', '2016-03-28'] ):
            target_list = feature_fun.getTarget(row)
            lag2[cust_id]= target_list[:]
            continue

        if (row['fecha_dato'] in ['2015-02-28', '2016-02-28'] ):
            target_list = feature_fun.getTarget(row)
            lag3[cust_id]= target_list[:]
            continue

        if (row['fecha_dato'] in ['2015-05-28', '2016-05-28'] ):
            target_list = feature_fun.getTarget(row)
            cust_dict[cust_id] =  target_list[:]
            total_dict[cust_id] = [feature_fun.getTotalTarget(row)]
            continue

        x_vars = []
        for col in feature_fun.cat_cols:
            x_vars.append(feature_fun.getIndex(row, col) )
        sex = feature_fun.getIndex(row, 'sexo')
        age = feature_fun.getAge(row)
        x_vars.append(age)
        x_vars.append(feature_fun.getMonth(row))
        x_vars.append(feature_fun.getjoinMonth(row))
        x_vars.append(feature_fun.getCustSeniority(row))
        income = feature_fun.getRent(row)
        x_vars.append(income)
        x_vars.append(feature_fun.getMarriageIndex(row, age, sex, income) )
        x_vars.append(feature_fun.getAge_group(row))
        x_vars.append(feature_fun.getCustSeniority_group(row))
        x_vars.append(feature_fun.sex_activity(row))
        x_vars.append(feature_fun.income_cate(row))
        x_vars.append(feature_fun.new_cust(row))

        if row['fecha_dato'] == '2016-06-28':
            prev_target_list = cust_dict.get(cust_id, [0]*22)
            lag_target_list = lag_cust_dict.get(cust_id, [0]*22)
            lag_target_list1 = lag1.get(cust_id, [0]*22)
            lag_target_list2 = lag2.get(cust_id, [0]*22)
            lag_target_list3 = lag3.get(cust_id, [0]*22)
            lag_target_list12 = lag12.get(cust_id, [0]*22)

            lag_feature_list = lag_feature.get(cust_id, [0.3162,0.0679,13,0,2,8,0,5,13,3,4,8])

            total_list = total_dict.get(cust_id, [0])
            x_vars_list.append(x_vars + lag_feature_list + prev_target_list + lag_target_list+total_list+lag_target_list1+lag_target_list2+
                               lag_target_list3+lag_target_list12)
        elif row['fecha_dato'] == '2015-06-28':
            prev_target_list = cust_dict.get(cust_id, [0]*22)
            lag_target_list = lag_cust_dict.get(cust_id, [0]*22)
            lag_target_list1 = lag1.get(cust_id, [0]*22)
            lag_target_list2 = lag2.get(cust_id, [0]*22)
            lag_target_list3 = lag3.get(cust_id, [0]*22)
            lag_target_list12 = lag12.get(cust_id, [0]*22)

            lag_feature_list = lag_feature.get(cust_id, [0.3162,0.0679,13,0,2,8,0,5,13,3,4,8])

            total_list = total_dict.get(cust_id, [0])
            target_list = feature_fun.getTarget(row)
            new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]
            if sum(new_products) > 0:
                for ind, prod in enumerate(new_products):
                    if prod>0:
                        assert len(prev_target_list) == 22
                        x_vars_list.append(x_vars+lag_feature_list+prev_target_list+lag_target_list+total_list+lag_target_list1+
                                           lag_target_list2 + lag_target_list3+lag_target_list12)
                        y_vars_list.append(ind)


    return x_vars_list, y_vars_list, cust_dict, lag_cust_dict,total_dict,lag1,lag2,lag3,lag12,lag_feature

if __name__ == "__main__":

    data_path = "/.../Santander/data"
    train_file =  open(data_path + "train.csv")
    x_vars_list, y_vars_list, cust_dict, lag_cust_dict,total_dict,lag1,lag2,lag3,lag12,lag_feature = \
        processDataMK(train_file, {}, {},{},{},{},{},{},{})

    with open(data_path +"x_train.pkl", "wb") as f:
                    cPickle.dump(x_vars_list, f, -1)
    with open(data_path +"y_train.pkl", "wb") as f:
                    cPickle.dump(y_vars_list, f, -1)
    with open(data_path +"cust_dict.pkl", "wb") as f:
                    cPickle.dump(cust_dict, f, -1)

    test_file = open(data_path + "test.csv")
    x_vars_list, y_vars_list, cust_dict, lag_cust_dict,total_dict,lag1,lag2, lag3,lag12,lag_feature = \
        processDataMK(test_file, cust_dict, lag_cust_dict,total_dict,lag1,lag2,lag3,lag12,lag_feature)

    with open(data_path + "x_test.pkl", "wb") as f:
                    cPickle.dump(x_vars_list, f, -1)
