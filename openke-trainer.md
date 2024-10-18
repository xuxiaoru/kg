import config
import numpy as np

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/FB15K/")

con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via torch.save() automatically.
con.set_export_files("./res/model.vec.pt")
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()


import config
import models
import numpy as np
import json

con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_flag(True)
con.set_work_threads(4)
con.set_dimension(100)
con.set_import_files("./res/model.vec.pt")
con.init()
con.set_model(models.TransE)
icon.test()



import json
import numpy as py

import config
import models
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_flag(True)
con.set_work_threads(4)
con.set_dimension(100)
con.set_import_files("./res/model.vec.pt")
con.init()
con.set_model(models.TransE)
# Get the embeddings (numpy.array)
embeddings = con.get_parameters("numpy")
# Get the embeddings (python list)
embeddings = con.get_parameters()



import json
import numpy as py
import config
import models
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_flag(True)
con.set_work_threads(4)
con.set_dimension(100)
con.init()
con.set_model(models.TransE)
f = open("./res/embedding.vec.json", "r")
embeddings = json.loads(f.read())
f.close()