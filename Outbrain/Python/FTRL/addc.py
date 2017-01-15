from collections import defaultdict
import marshal

f_train = "/Users/xiaofeifei/I/Kaggle/train"

o_train = "../train_c"



def get_id(id,ip,device):
    if id != "i_a99f214a":
        return id
    else:
        return ip + "_" + device

def run(input,output,isTest):
	f = open(input)
	o = open(output,"w")
	d = defaultdict(int)
	d2 = {}
	dh = defaultdict(int)
	line = f.readline()
	print line[:-2]
	print >> o,line[:-2] + ",C22,C23,C24,C25,C26,C27,C28"
	count = 0
	day = "??"
	hour = "??"
	date_idx = 2
	if isTest:
		date_idx = 1
	while True:
		line = f.readline()
		if not line:
			break
		count += 1
		if count > 2:
			break
		lis = line.split(",")
		print lis[date_idx]
		print lis[date_idx][4:6]
		if lis[date_idx][4:6] != day:
			del d
			d = defaultdict(int)
			d2 = {}
			day = lis[date_idx][4:6]
			print lis
			print lis[2+9]


		if lis[date_idx][6:] != hour:
			del dh
			dh = defaultdict(int)
			hour = lis[date_idx][6:]
		time = int(lis[date_idx][6:]) * 60 + int(int(lis[0][:5]) / 100000. * 60)
		id = get_id("i_"+lis[date_idx+9],"j_"+lis[date_idx+10],"k_"+lis[date_idx+11])
		d[id + "_n_" + lis[date_idx+14]] += 1
		d[id + "_q_" + lis[date_idx+17]] += 1
		dh[id + "_n_" + lis[date_idx+14]] += 1
		dh[id + "_q_" + lis[date_idx+17]] += 1
		dh[id] += 1

		media_id = "f_"+lis[date_idx+6]
		if lis[date_idx+6] == "ecad2386":
			media_id = "c_"+lis[date_idx+3]
		d[id + "_" + media_id] += 1
		t = "-1"

		if id not in d2:
			d2[id] = time
		else:
			t = str(time-d2[id])
			d2[id] = time

		m = d[id + "_" + media_id]
		c = d[id + "_n_" + lis[date_idx+14]]
		c2 = d[id + "_q_" + lis[date_idx+17]]
		ch = dh[id + "_n_" + lis[date_idx+14]]
		ch1 = dh[id + "_q_" + lis[date_idx+17]]
		ch2 = dh[id]
		print >> o,line[:-2] + "," + id + "," + str(m) + "," + str(ch1) + "," + str(ch2) + "," +  str(c) + "," + str(c2) + "," + t
	f.close()
	o.close()

run(f_train,o_train,False)