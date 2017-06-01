import csv
from csv import DictReader


data_path = "/Users/xiaofeifei/I/Kaggle/Outbrain/"

def process_data(path, D,prcont_dict,prcont_header,event_dict,event_header,leak_uuid_dict,document_header):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        disp_id = int(row['display_id'])
        ad_id = int(row['ad_id'])

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        x = []
        for key in row:
            x.append(abs(hash(key + '_' + row[key])) % D)

        row = prcont_dict.get(ad_id, [])
        # build x
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

        # document_id = int(row[0])
        # row = event_dict.get(disp_id, [])

        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            x.append(abs(hash(event_header[ind] + '_' + val)) % D)

        # row = document_dict.get(document_id, [0,0,0])
        # source_id = row[0]

        for ind, val in enumerate(row):
            x.append(abs(hash(document_header[ind] + '_' + val)) % D)
        #
        # # cate
        # row = document_cate_dict.get(document_id, [])
        #
        # for ind, val in enumerate(row):
        #     x.append(abs(hash(document_cate_header[ind] + '_' + val)) % D)
        #
        # # entities
        # row = document_en_dict.get(document_id, [])
        #
        # for ind, val in enumerate(row):
        #     x.append(abs(hash(document_en_header[ind] + '_' + val)) % D)
        #
        # # topics
        # row = document_top_dict.get(document_id, [])
        #
        # for ind, val in enumerate(row):
        #     x.append(abs(hash(document_top_header[ind] + '_' + val)) % D)

        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):
            x.append(abs(hash('leakage_row_found_1'))%D)
        else:
            x.append(abs(hash('leakage_row_not_found'))%D)

        iter = len(x)
        for i in xrange(iter):
            x.append(abs(hash(str(x[i]) + '_' + str(ad_id))) % D)
            # x.append(abs(hash(str(x[i]) + '_' + str(source_id))) % D)

        yield t, disp_id, ad_id, x, y

